import os
from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.tensorboard import SummaryWriter
import torch
from uni_world_model.data.datasets import DataPrefetcher
import omegaconf
from glob import glob
import shutil
from collections import defaultdict
from torch.distributions import Beta
from uni_world_model.diffusion_policy.trainers.trainer_utils import cross_entropy, masked_loss, visualize_latent_action_gen

class DP_Trainer(nn.Module):
    def __init__(
        self,
        diffusion_policy,
        diffusion_policy_config,
        latent_action_tokenizer,
        rgb_preprocessor,
        lang_tokenizer,
        train_dataloader,
        eval_dataloader,
        save_path,
        save_epochs=1,
        save_steps=10000,
        num_epochs=20,
        print_steps=100,
        lr_max=0.0001,
        weight_decay=0.0001,
        num_warmup_epochs=1,
        gradient_accumulation_steps=4,
        resume_ckpt_path=None,
        bs_per_gpu=32,
        max_epoch=None,
        pred_binary_gripper_action=True,
    ):
        super(DP_Trainer, self).__init__()
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator= Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            kwargs_handlers=[ddp_kwargs]
        )
        self.accelerator = accelerator

        if resume_ckpt_path is not None:
            self.print(f"resuming WorldModel from {resume_ckpt_path} ...")

            current_model_dict = diffusion_policy.state_dict()
            resume_model_dict = torch.load(os.path.join(resume_ckpt_path, 'pytorch_model.bin'), map_location='cpu')

            mismatched_param_names = []
            filtered_state_dict = {}

            for name, param in resume_model_dict.items():
                if name in current_model_dict and current_model_dict[name].shape != param.shape:
                    mismatched_param_names.append(name)
                else:
                    filtered_state_dict[name] = param

            missing_keys, unexpected_keys = diffusion_policy.load_state_dict(filtered_state_dict, strict=False)
            missing_root_keys = set([k.split(".")[0] for k in missing_keys])
            self.print('load ', resume_ckpt_path, '\nmissing ', missing_root_keys, '\nunexpected ', unexpected_keys, '\nmismatched ', mismatched_param_names)
        
        
        optimizer = torch.optim.AdamW(diffusion_policy.parameters(), lr=lr_max, weight_decay=weight_decay, fused=True)
        total_prints_per_epoch = len(train_dataloader.dataset) // (print_steps * bs_per_gpu * accelerator.num_processes)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=min(num_warmup_epochs*total_prints_per_epoch, 5000000 // (print_steps * bs_per_gpu * accelerator.num_processes)),
            num_training_steps=num_epochs*total_prints_per_epoch,
        )
        diffusion_policy, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            diffusion_policy, optimizer, train_dataloader, eval_dataloader, 
            device_placement=[True, True, False, False]
        )
        if latent_action_tokenizer is not None:
            latent_action_tokenizer = latent_action_tokenizer.to(accelerator.device)
            latent_action_tokenizer.eval()
        
        self.writer = SummaryWriter(os.path.join(save_path, 'logs'))
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.total_prints_per_epoch = total_prints_per_epoch
        self.diffusion_policy = diffusion_policy
        self.diffusion_policy_config = diffusion_policy_config
        self.latent_action_tokenizer = latent_action_tokenizer
        self.optimizer = optimizer
        self.train_prefetcher = DataPrefetcher(train_dataloader, self.device, lang_tokenizer=lang_tokenizer)
        self.eval_prefetcher = DataPrefetcher(eval_dataloader, self.device, lang_tokenizer=lang_tokenizer)
        self.rgb_preprocessor = rgb_preprocessor.to(self.device)
        self.lang_tokenizer = lang_tokenizer
        self.save_path = save_path
        self.save_epochs = save_epochs
        self.save_steps = save_steps
        self.max_epoch = max_epoch
        self.num_epochs = num_epochs
        self.print_steps = print_steps
        self.bs_per_gpu = bs_per_gpu
        self.pred_binary_gripper_action = pred_binary_gripper_action

        self.beta_dist = Beta(self.diffusion_policy_config.noise_beta_alpha, self.diffusion_policy_config.noise_beta_beta)
        self.num_timestep_buckets = self.diffusion_policy_config.num_timestep_buckets


    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def process_index(self):
        return self.accelerator.process_index

    def print(self, *args, **kwargs):
        self.accelerator.print(*args, **kwargs)

    def save_checkpoint(self, save_dir):
        unwrapped_diffusion_policy = self.accelerator.unwrap_model(self.diffusion_policy)
        state_dict = unwrapped_diffusion_policy.get_state_dict_to_save()
        
        torch.save(state_dict, os.path.join(save_dir, "pytorch_model.bin"))
        omegaconf.OmegaConf.save(unwrapped_diffusion_policy.config, os.path.join(save_dir, "config.yaml"))
        self.print(f"A new model checkpoint is saved to {save_dir}!!!")

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.diffusion_policy_config.noise_s - sample) / self.diffusion_policy_config.noise_s
        
    def train(self):
        step = 0
        
        for epoch in range(self.num_epochs+1):
            if epoch != 0:
                self.accelerator.wait_for_everyone()
                save_dir = os.path.join(self.save_path, f'saved_epoch_{epoch}_step_{step}')

                if self.is_main:
                    os.makedirs(save_dir, exist_ok=True)
                    self.save_checkpoint(save_dir)

                if self.diffusion_policy_config.world_model.latent_action_pred:
                    visualization_dir = os.path.join(save_dir, 'visualization')
                    self.eval_latent_action_gen(visualization_dir)

                if epoch == self.num_epochs:
                    break
                if (self.max_epoch is not None) and (epoch >= self.max_epoch):
                    break

            log_loss = {
                'latent_action_loss': torch.tensor(0).float().to(self.device),
                'action_loss': torch.tensor(0).float().to(self.device),
                'total_loss': torch.tensor(0).float().to(self.device),
            }
            eval_log_loss = {
                'latent_action_loss': torch.tensor(0).float().to(self.device),
                'action_loss': torch.tensor(0).float().to(self.device),
                'total_loss': torch.tensor(0).float().to(self.device),
            }
            
            cum_load_time = 0 
            clock = time()
            batch_idx = 0
            batch, load_time = self.train_prefetcher.next()
            
            while batch is not None:
                with self.accelerator.accumulate(self.diffusion_policy):

                    self.diffusion_policy.train()
                    self.optimizer.zero_grad()
                    loss = self.calculate_loss(batch, train=True)
                    self.accelerator.backward(loss['total_loss'])
                    self.optimizer.step()

                    for key in log_loss:
                        log_loss[key] += loss[key].detach() / self.print_steps
                    cum_load_time += load_time / self.print_steps

                if (batch_idx+1) % self.print_steps == 0:

                    with torch.no_grad():
                        self.diffusion_policy.eval()
                        batch, _ = self.eval_prefetcher.next_without_none()
                        loss = self.calculate_loss(batch, train=True)
                        for key in eval_log_loss:
                            eval_log_loss[key] = loss[key].detach()

                    self.log(log_loss, eval_log_loss, cum_load_time, clock, epoch, batch_idx, step)
                    for key in log_loss:
                        log_loss[key] = torch.tensor(0).float().to(self.device)
                    for key in eval_log_loss:
                        eval_log_loss[key] = torch.tensor(0).float().to(self.device)

                    cum_load_time = 0
                    clock = time()
                    self.scheduler.step()

                if batch_idx  % self.save_steps == 0: #(batch_idx+1)  % self.save_steps == 0:
                    self.accelerator.wait_for_everyone()
                    save_dir = os.path.join(self.save_path, f'temp_epoch_{epoch}_step_{step}')

                    if self.is_main:
                        existing_ckpt_dirs = glob(os.path.join(self.save_path, f'temp_epoch_*_step_*'))
                        for existing_ckpt_dir in existing_ckpt_dirs:
                            if existing_ckpt_dir != save_dir:
                                shutil.rmtree(existing_ckpt_dir)
                        os.makedirs(save_dir, exist_ok=True)
                        self.save_checkpoint(save_dir)

                    if self.diffusion_policy_config.world_model.latent_action_pred:
                        visualization_dir = os.path.join(save_dir, 'visualization')
                        self.eval_latent_action_gen(visualization_dir)
        

                batch_idx += 1
                step += 1
                batch, load_time = self.train_prefetcher.next()



    @torch.no_grad()
    def eval_latent_action_gen(self, visualization_dir):
        pass

    def calculate_loss(self, batch, train):
        # image preprocessing
        if not self.diffusion_policy_config.freeze_world_model:
            rgb_seq = torch.cat([batch['rgb_initial'], batch['rgb_future'][:,:1]], dim=1)
            rgb_seq = self.rgb_preprocessor(rgb_seq, train=train)
            rgb_initial = rgb_seq[:,:1]
        else:
            rgb_initial = self.rgb_preprocessor(batch['rgb_initial'], train=train)

        # obtain ground-truth latent action ids
        if not self.diffusion_policy_config.freeze_world_model:
            # b, t, c, h, w = batch['rgb_future'].shape
            b, t, c, h, w = rgb_seq.shape
            t = t - 1 # t=1
            latent_action_ids = self.latent_action_tokenizer(
                cond_pixel_values=rgb_seq[:,:-1].reshape(-1, c, h, w),
                target_pixel_values=rgb_seq[:,1:].reshape(-1, c, h, w),
                return_action_token_ids_only=True
            ).reshape(b, -1)
        else:
            latent_action_ids = None

        # Embed noised action trajectory.
        # (b, action_chunk, action_dim)
        actions = batch['actions'][:,0]
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        
        # compute loss
        pred = self.diffusion_policy(
            rgb=rgb_initial, # (b, 1, c, h, w)
            language=batch['lang_input_ids'],
            latent_action_ids=latent_action_ids, # (b, per_latent_action_len)
            noisy_trajectory=noisy_trajectory,
            t_discretized=t_discretized,
            train=True,
        ) 

        latent_action_preds = pred['latent_action_preds']
        pred_v = pred['pred_v']        

        b, action_chunk, action_dim = actions.shape
        velocity = velocity.reshape(b, 1, action_chunk, action_dim)
        pred_v = pred_v.reshape(b, 1, action_chunk, action_dim)
        loss = {}
        latent_mask = batch['latent_mask']
        mask = batch['mask']
        loss['latent_action_loss'] = masked_loss(latent_action_preds, latent_action_ids, latent_mask, 0, cross_entropy) if latent_action_preds is not None else torch.tensor(0.0).to(actions.device)
        loss['action_loss'] = masked_loss(pred_v, velocity, mask, 0, F.mse_loss) if pred_v is not None else torch.tensor(0.0).to(actions.device)
        total_loss = loss['latent_action_loss'] + 200 * loss['action_loss']
        loss['total_loss'] = total_loss

        return loss

    def log(self, log_loss, eval_log_loss, cum_load_time, clock, epoch, batch_idx, step):
        for key in log_loss:
            log_loss[key] = self.accelerator.gather_for_metrics(log_loss[key]).mean()
        for key in eval_log_loss:
            eval_log_loss[key] = self.accelerator.gather_for_metrics(eval_log_loss[key]).mean()
        load_pecnt = torch.tensor(cum_load_time / (time()-clock)).to(self.device)
        load_pecnt = self.accelerator.gather_for_metrics(load_pecnt).mean()
        fps = (self.bs_per_gpu*self.print_steps*(self.diffusion_policy_config.world_model.sequence_length+1)) / (time()-clock)
        fps = self.accelerator.gather_for_metrics(torch.tensor(fps).to(self.device)).sum()

        text = 'Train Epoch: {} [{}/{} ({:.0f}%)] FPS:{:.5f} Load Pertentage:{:.5f} LR:{}'.format(
            epoch, 
            batch_idx * self.bs_per_gpu * self.accelerator.num_processes, 
            len(self.train_prefetcher), 
            100. * batch_idx * self.bs_per_gpu * self.accelerator.num_processes / len(self.train_prefetcher),
            fps,
            load_pecnt,
            self.scheduler.get_last_lr()[0],
        )
        for key in log_loss:
            text = text + ' {}_loss: {:.5f}'.format(key, log_loss[key])
        for key in eval_log_loss:
            text = text + ' eval_{}_loss: {:.5f}'.format(key, eval_log_loss[key])
        self.print(text)
        if self.is_main:
            for key in log_loss:
                self.writer.add_scalar(key+'_loss', log_loss[key], step)
            for key in eval_log_loss:
                self.writer.add_scalar('eval_'+key+'_loss', eval_log_loss[key], step)
            self.writer.add_scalar("learning rate", self.scheduler.get_last_lr()[0], step)
            self.writer.add_scalar("FPS", fps, step)
            self.writer.add_scalar("loading time in total time", load_pecnt, step)
