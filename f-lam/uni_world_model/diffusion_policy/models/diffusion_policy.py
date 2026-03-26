# Copyright (2024) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn as nn
import torch.nn.functional as F
from uni_world_model.diffusion_policy.modules.action import ActionEncoder, ActionDecoder
from IPython import embed

class DiffusionPolicyModel(nn.Module):
    def __init__(
            self,
            world_model,
            diffusion_head,
            freeze_world_model,
            pretrained_model_path,
            dp_hidden_size,
            input_embedding_dim,
            noise_beta_alpha,
            noise_beta_beta,
            noise_s,
            num_inference_timesteps,
            num_timestep_buckets,
            act_pred,
            act_dim,
            add_pos_embed,
            **kwargs
    ):
        super(DiffusionPolicyModel, self).__init__()
        self.world_model = world_model
        self.diffusion_head = diffusion_head
        self.freeze_world_model = freeze_world_model
        self.pretrained_model_path = pretrained_model_path
        self.dp_hidden_size = dp_hidden_size
        self.input_embedding_dim = input_embedding_dim
        self.noise_beta_alpha = noise_beta_alpha
        self.noise_beta_beta = noise_beta_beta
        self.noise_s = noise_s
        self.num_inference_timesteps = num_inference_timesteps
        self.num_timestep_buckets = num_timestep_buckets
        self.act_pred = act_pred
        self.act_dim = act_dim
        self.add_pos_embed = add_pos_embed

        self.action_encoder = ActionEncoder(
            action_dim=self.act_dim,
            hidden_size=self.input_embedding_dim,
        )
        self.action_decoder = ActionDecoder(
            input_dim=self.dp_hidden_size,
            hidden_dim=self.dp_hidden_size,
            output_dim=self.act_dim,
        )

        if self.add_pos_embed:
            max_seq_len = 1024
            self.position_embedding = nn.Embedding(max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        ## load pretrained world model
        if self.pretrained_model_path:
            missing_keys, unexpected_keys = self.world_model.load_state_dict(torch.load(self.pretrained_model_path), strict=False)
            missing_root_keys = set([k.split(".")[0] for k in missing_keys])
            print('load ', self.pretrained_model_path, '\nmissing ', missing_root_keys, '\nunexpected ', unexpected_keys)

            if self.freeze_world_model:
                self.world_model.eval()
                # Freeze the model parameters
                for k, p in self.world_model.named_parameters():
                    p.requires_grad_(False)
        
    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config['noise_s'] - sample) / self.config['noise_s']

    def forward(self, 
                rgb, # (b, 1, c, h, w)
                language,
                latent_action_ids, # (b, per_latent_action_len)
                noisy_trajectory,
                t_discretized,
                train=True,
                **kwargs
    ):
        if train:
            latent_action_pred = self.world_model(rgb, language, latent_action_ids, train=train)
            latent_action_feature = latent_action_pred['latent_action_feature'] # (b, per_latent_action_len, h)
            cond_input_feature = latent_action_pred['cond_input_feature'] # (b, n_cond_tokens, h)
            latent_action_preds = latent_action_pred['latent_action_preds'] # (b, per_latent_action_len, latent_action_codebook_size)
        else:
            latent_action_feature = latent_action_ids['latent_action_feature'] # (b, per_latent_action_len, h)
            cond_input_feature = latent_action_ids['cond_input_feature'] # (b, n_cond_tokens, h)
            latent_action_preds = latent_action_ids['latent_action_preds'] # (b, per_latent_action_len, latent_action_codebook_size)

        action_features = self.action_encoder(noisy_trajectory, t_discretized)

        if self.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=noisy_trajectory.device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        sa_embs = action_features

        vl_embs = torch.cat((latent_action_feature, cond_input_feature), dim=1)

        model_output = self.diffusion_head(
            hidden_states=sa_embs, # (b, action_chunk, h)
            encoder_hidden_states=vl_embs, # (b, n_cond_tokens+per_latent_action_len, h)
            encoder_attention_mask=None,
            timestep=t_discretized,
        )
        pred = self.action_decoder(model_output)
        pred_v = pred[:, -noisy_trajectory.shape[1] :] # (b, action_chunk, action_dim)

        res = {}
        res['latent_action_preds'] = latent_action_preds
        res['pred_v'] = pred_v
        return res


    @property
    def device(self):
        return next(self.parameters()).device

    def get_state_dict_to_save(self):
        modules_to_exclude = ['model_lang', 'model_vision']
        state_dict = {k: v for k, v in self.state_dict().items() if
                      not any(module_name in k for module_name in modules_to_exclude)}
        return state_dict
