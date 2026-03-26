from typing import Dict, List
import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat
from uni_world_model.latent_action_model.modules.blocks import patchify, unpatchify, SpatioTemporalTransformer, SpatioTransformer, \
                                                     MVSpatioTemporalTransformer, MVSpatioTransformer
from uni_world_model.latent_action_model.modules.vector_quantizer import VectorQuantizer, VectorQuantizer2
from IPython import embed
from torchvision import transforms

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TORCH_HOME"] = "/mnt/workspace/czj/models"

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def edge_aware_log_l1_loss(pred, gt, rgb):
    """Gradient-aware Log-L1 depth loss (EdgeAwareLogL1).

    Args:
        pred: predicted depth, shape [N, C, H, W]
        gt:   ground-truth depth, shape [N, C, H, W]
        rgb:  corresponding RGB image, shape [N, 3, H, W]
    Returns:
        scalar loss
    """
    logl1 = torch.log(1 + torch.abs(pred - gt))  # per-pixel Log-L1

    grad_img_x = torch.mean(torch.abs(rgb[:, :, :, :-1] - rgb[:, :, :, 1:]), dim=1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(rgb[:, :, :-1, :] - rgb[:, :, 1:, :]), dim=1, keepdim=True)
    lambda_x = torch.exp(-grad_img_x)
    lambda_y = torch.exp(-grad_img_y)

    loss_x = (lambda_x * logl1[:, :, :, :-1]).mean()
    loss_y = (lambda_y * logl1[:, :, :-1, :]).mean()
    return loss_x + loss_y

class LatentActionTokenizer(nn.Module):
    """
    Latent action VQ-VAE.
    """

    def __init__(
            self,
            in_dim: int,
            model_dim: int,
            latent_dim: int,
            num_latents: int,
            patch_size: int,
            enc_blocks: int,
            dec_blocks: int,
            num_heads: int,
            action_num_codes: int,
            num_frame_pred: int,
            loss_config,
            dropout: float = 0.0,
    ) -> None:
        super(LatentActionTokenizer, self).__init__()
        self.dino_transform = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        self.dino_encoder = torch.hub.load('/mnt/workspace/czj/models/dinov2', 'dinov2_vitb14_reg', source='local')
        self.dino_encoder.requires_grad_(False)

        dino_dim = 768
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        patch_token_dim = in_dim * patch_size ** 2
        patches_depth_token_dim = 1 * patch_size ** 2
        self.num_codes = action_num_codes
        self.num_frame_pred = num_frame_pred
        self.action_latent = nn.Parameter(torch.empty(1, self.num_frame_pred+1, self.num_codes, dino_dim))    # TODO: num of codes
        nn.init.uniform_(self.action_latent, a=-1, b=1)
        self.encoder = SpatioTemporalTransformer(
            in_dim=dino_dim,
            model_dim=model_dim,
            out_dim=latent_dim,
            num_blocks=enc_blocks,
            num_heads=num_heads,
            dropout=dropout,
            causal_temporal=True,
            to_out=False,
        )

        self.to_codebook = nn.Linear(model_dim, latent_dim)
        # self.vector_quantizer = VectorQuantizer(
        #     num_latents=num_latents,
        #     latent_dim=latent_dim,
        #     code_restart=True,
        # )

        self.vector_quantizer = VectorQuantizer2(
            num_latents=num_latents,
            latent_dim=latent_dim,
            beta=0.25,
            remap=None,
            sane_index_shape=True,
        )
        ## Decoder: Spatial Transformer
        self.patch_up = nn.Linear(patch_token_dim, model_dim)
        self.patch_up_depth = nn.Linear(patches_depth_token_dim, model_dim)
        self.action_up = nn.Linear(latent_dim, model_dim)
        self.decoder = SpatioTransformer(
            in_dim=model_dim,
            model_dim=model_dim,
            out_dim=patch_token_dim,       
            num_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.timestep_embedding = nn.Parameter(torch.empty(self.num_frame_pred, 1, model_dim))
        nn.init.uniform_(self.timestep_embedding, a=-1, b=1)
        self.loss_config = loss_config
        self.loss_fn_lpips = lpips.LPIPS(net='vgg').requires_grad_(False).eval()



    def vq_encode(self, videos: Tensor, attention_mask: Tensor = None) -> Dict:
        # Preprocess videos
        B, T = videos.shape[:2]
        patches = patchify(videos, self.patch_size)
        videos = rearrange(videos, "b T c h w -> (b T) c h w")
        videos = self.dino_transform(videos)
        dino_features = self.dino_encoder.forward_features(videos)['x_norm_patchtokens']
        dino_features = rearrange(dino_features, "(b T) l d -> b T l d", b = B, T = T)

        action_pad = self.action_latent.expand(B, -1, -1, -1)
        padded_patches = torch.cat([action_pad, dino_features], dim=2)

        # Encode
        # full attention
        z = self.encoder(padded_patches)
      
        # Get latent action for all future frames
        z = self.to_codebook(z[:, 1:, :self.num_codes])  # (B, T-1, n, E)

        # Vector quantize
        z = z.reshape(B * (T - 1), self.num_codes, self.latent_dim)
        z_q, indices, commit_loss  = self.vector_quantizer(z)
        indices = indices.reshape(B, T - 1, self.num_codes)
        z_q = z_q.reshape(B, T - 1, self.num_codes, self.latent_dim)
        return {
            "dino_features": dino_features,
            "patches": patches,
            "z_q": z_q,
            "indices": indices,
            "commit_loss": commit_loss,
        }

    def forward(self, cond_pixel_values, target_pixel_values, return_action_token_ids_only=False, return_recons_only=False) -> Dict:
        video = torch.cat([cond_pixel_values, target_pixel_values], dim=1)
        # Encode + VQ
        video_depth = video[:,:,3:]
        video = video[:,:,:3]
        B, T = video.shape[:2]
        H, W = video.shape[3:5]
        
        patches_depth = patchify(video_depth, self.patch_size)
        outputs = self.vq_encode(video) 
        if return_action_token_ids_only:
            return outputs['indices']
        video_patches = self.patch_up(outputs["patches"][:, :1]) # (B, 1, l, d)
        depth_patches = self.patch_up_depth(patches_depth[:, :1]) # (B, 1, l, d)
        video_patches = video_patches.repeat(1, T-1, 1, 1) # (B, T-1, l, d)
        depth_patches = depth_patches.repeat(1, T-1, 1, 1)
        action_patches = self.action_up(outputs["z_q"])    # (B, T-1, n, d)
        timestep_embedding = self.timestep_embedding.repeat(B, 1, 1, 1)
        video_action_patches = torch.cat([action_patches, video_patches, depth_patches, timestep_embedding], dim=2) # (B, T-1, n+l, d)

        # Decode
        # mask future frames
        video_recon_all = self.decoder(video_action_patches) # (B, T-1, n+l, p)
        video_recon = video_recon_all[:, :, self.num_codes: self.num_codes + video_patches.shape[2]]
        depth_recon = video_recon_all[:, :, self.num_codes + video_patches.shape[2]: self.num_codes + video_patches.shape[2] + depth_patches.shape[2]]
        video_recon = F.sigmoid(video_recon)
        depth_recon = F.relu(depth_recon)

        target_video_recon = unpatchify(video_recon, self.patch_size, H, W) # (B, T-1, C, H, W)
        target_depth_recon = unpatchify(depth_recon, self.patch_size, H, W)
        # recon_pixel_values = target_video_recon.squeeze(1)
        recon_pixel_values = target_video_recon
        recon_depth_values = target_depth_recon
        
        if return_recons_only:
            return {
                "recon_pixel_values": recon_pixel_values,
                "recon_depth_values": recon_depth_values,
                "indices": outputs['indices'],
            }
        
        # Compute loss
        recon_pixel_values = rearrange(recon_pixel_values, 'b t c h w -> (b t) c h w')
        target_pixel_values = rearrange(target_pixel_values[:,:,:3], 'b t c h w -> (b t) c h w')
        recon_depth_values = rearrange(target_depth_recon, 'b t c h w -> (b t) c h w')
        target_depth_values = rearrange(video_depth[:,1:], 'b t c h w -> (b t) c h w')

        depth_loss = torch.abs(recon_depth_values - target_depth_values).mean()

        # EdgeAwareLogL1: gradient-aware Log-L1 weighted by RGB edges
        # After unfold/flatten, shapes are [B, 1, H, W]; repeat rgb accordingly
        n_depth = recon_depth_values.shape[0]
        target_pixel_values_depth = target_pixel_values
        n_rgb = target_pixel_values.shape[0]
        if n_depth != n_rgb:
            repeat_factor = n_depth // n_rgb
            target_pixel_values_depth = target_pixel_values.repeat_interleave(repeat_factor, dim=0)
        depth_h, depth_w = recon_depth_values.shape[-2], recon_depth_values.shape[-1]
        target_pixel_values_depth = F.interpolate(target_pixel_values_depth, size=(depth_h, depth_w), mode='bilinear', align_corners=False)
        loss_pred_depth_x = edge_aware_log_l1_loss(recon_depth_values, target_depth_values[:, 0], target_pixel_values_depth)

        if self.loss_config.use_abs_recons_loss:
            recons_loss = torch.abs(recon_pixel_values - target_pixel_values).mean()
        else:
            recons_loss = F.mse_loss(recon_pixel_values, target_pixel_values)

        if self.loss_config.perceptual_loss_w > 0:
            with torch.no_grad():
                perceptual_loss = self.loss_fn_lpips.forward(
                    recon_pixel_values, target_pixel_values, normalize=True).mean()
        else:
            perceptual_loss = torch.zeros_like(recons_loss)

        commit_loss = outputs['commit_loss']
        loss =  self.loss_config.commit_loss_w * commit_loss + self.loss_config.recon_loss_w * recons_loss + \
                self.loss_config.perceptual_loss_w * perceptual_loss + depth_loss * 0.1
        
        # active_code_num = torch.tensor(len(set(indices.long().reshape(-1).cpu().numpy().tolist()))).float().to(loss.device)
        active_code_num = torch.tensor(torch.unique(outputs['indices']).shape[0]).float().to(loss.device)

        loss_outputs = dict()
        loss_outputs.update(
            {
                "active_code_num": active_code_num,
                "loss": loss,
                "commit_loss": commit_loss,
                "recons_loss": recons_loss,
                "depth_loss": depth_loss,
                "perceptual_loss": perceptual_loss,
            }
        )
        return loss_outputs
    
    @torch.no_grad()
    def decode_image(self, cond_pixel_values, given_action_token_ids):
        B, _, _, H, W = cond_pixel_values.shape
        given_action_token_ids = given_action_token_ids.reshape(B*self.num_frame_pred, self.num_codes)
        patches = patchify(cond_pixel_values, self.patch_size)
        video_patches = self.patch_up(patches)  # (B, 1, num_patches, model_dim)
        video_patches = video_patches.repeat(1, self.num_frame_pred, 1, 1) # (B, T-1, l, d)

        z_q = self.vector_quantizer.codebook(given_action_token_ids, shape=(B*self.num_frame_pred, self.num_codes, self.latent_dim))
        z_q = z_q.reshape(B, self.num_frame_pred, self.num_codes, self.latent_dim)
        action_patches = self.action_up(z_q)    # (B, T-1, n, d)
        timestep_embedding = self.timestep_embedding.repeat(B, 1, 1, 1)
        video_action_patches = torch.cat([action_patches, video_patches, timestep_embedding], dim=2) # (B, T-1, n+l, d)

        # Decode
        # mask future frames
        video_recon = self.decoder(video_action_patches) # (B, T-1, n+l, p)
        video_recon = video_recon[:, :, self.num_codes: self.num_codes + video_patches.shape[2]] 
        video_recon = F.sigmoid(video_recon)

        target_video_recon = unpatchify(video_recon, self.patch_size, H, W) # (B, T-1, C, H, W)
        # recon_pixel_values = target_video_recon.squeeze(1)
        recon_pixel_values = target_video_recon

        return {
                "recon_pixel_values": recon_pixel_values,
            }


    @property
    def device(self):
        return next(self.parameters()).device

    def get_state_dict_to_save(self):
        modules_to_exclude = []
        state_dict = {k: v for k, v in self.state_dict().items() if
                      not any(module_name in k for module_name in modules_to_exclude)}
        return state_dict
