from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from uni_world_model.latent_action_model.modules.blocks import patchify, unpatchify, SpatioTemporalTransformer, SpatioTransformer
from torch import Tensor
import lpips


class LatentActionModel(nn.Module):
    """
    Latent action VAE.
    """

    def __init__(
            self,
            in_dim: int,
            model_dim: int,
            latent_dim: int,
            patch_size: int,
            enc_blocks: int,
            dec_blocks: int,
            num_heads: int,
            loss_config,
            training: bool = True,
            dropout: float = 0.0
    ) -> None:
        super(LatentActionModel, self).__init__()
        self.model_dim = model_dim
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        patch_token_dim = in_dim * patch_size ** 2

        self.action_prompt = nn.Parameter(torch.empty(1, 1, 1, patch_token_dim))
        nn.init.uniform_(self.action_prompt, a=-1, b=1)
        self.encoder = SpatioTemporalTransformer(
            in_dim=patch_token_dim,
            model_dim=model_dim,
            out_dim=model_dim,
            num_blocks=enc_blocks,
            num_heads=num_heads,
            dropout=dropout
        )
        self.fc = nn.Linear(model_dim, latent_dim * 2)
        self.patch_up = nn.Linear(patch_token_dim, model_dim)
        self.action_up = nn.Linear(latent_dim, model_dim)
        self.decoder = SpatioTransformer(
            in_dim=model_dim,
            model_dim=model_dim,
            out_dim=patch_token_dim,
            num_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout
        )
        
        import os
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        os.environ["TORCH_HOME"] = "/mnt/workspace/czj/models"
        self.loss_fn_lpips = lpips.LPIPS(net='vgg').requires_grad_(False).eval()

        self.mu_record = None
        self.training = True
        self.loss_config = loss_config

    def encode(self, videos: Tensor) -> Dict:
        # Preprocess videos
        B, T = videos.shape[:2]
        patches = patchify(videos, self.patch_size)

        action_pad = self.action_prompt.expand(B, T, -1, -1)
        padded_patches = torch.cat([action_pad, patches], dim=2)

        # Encode
        z = self.encoder(padded_patches)  # (B, T, 1+N, E)
        # Get latent action for all future frames
        z = z[:, 1:, 0]  # (B, T-1, 1, E)

        # VAE
        z = z.reshape(B * (T - 1), self.model_dim)
        moments = self.fc(z)
        z_mu, z_var = torch.chunk(moments, 2, dim=1)
        # Reparameterization
        if not self.training:
            z_rep = z_mu
        else:
            z_rep = z_mu + torch.randn_like(z_var) * torch.exp(0.5 * z_var)
        z_rep = z_rep.reshape(B, T - 1, 1, self.latent_dim)

        if not self.training:
            if self.mu_record is None:
                self.mu_record = z_mu
            else:
                self.mu_record = torch.cat([self.mu_record, z_mu], dim=0)

        return {
            "patches": patches,
            "z_rep": z_rep,
            "z_mu": z_mu,
            "z_var": z_var
        }

    def forward(self, cond_pixel_values, target_pixel_values, return_action_token_ids_only=False, return_recons_only=False):
    
        if return_action_token_ids_only or return_recons_only:
            self.training = False
        else:
            self.training = True

        video = torch.stack([cond_pixel_values, target_pixel_values], dim=1)
        B, T, C, H, W = video.shape

        outputs = self.encode(video)
        video_patches = self.patch_up(outputs["patches"][:, :-1]) # (B, T-1, num_patches, model_dim)
        action_patches = self.action_up(outputs["z_rep"]) # (B, T-1, 1, model_dim)
        video_action_patches = video_patches + action_patches # (B, T-1, num_patches, model_dim)

        # Decode
        video_recon = self.decoder(video_action_patches)
        video_recon = F.sigmoid(video_recon)
        recon_pixel_values = (unpatchify(video_recon, self.patch_size, H, W)).squeeze(1)
        
        # Compute loss
        if self.loss_config.use_abs_recons_loss:
            recons_loss = torch.abs(target_pixel_values - recon_pixel_values).mean()
        else:
            recons_loss = F.mse_loss(target_pixel_values, recon_pixel_values)

        if self.loss_config.perceptual_loss_w > 0:
            with torch.no_grad():
                perceptual_loss = self.loss_fn_lpips.forward(
                    recon_pixel_values, target_pixel_values, normalize=True).mean()
        else:
            perceptual_loss = torch.zeros_like(recons_loss)

        kl_loss = -0.5 * torch.sum(1 + outputs["z_var"] - outputs["z_mu"] ** 2 - outputs["z_var"].exp(), dim=1).mean()
        loss = self.loss_config.kl_loss_w * kl_loss + self.loss_config.recon_loss_w * recons_loss + self.loss_config.perceptual_loss_w * perceptual_loss
        
        outputs.update(
            {
                "recon_pixel_values": recon_pixel_values,
                "loss": loss,
                "kl_loss": kl_loss,
                "recons_loss": recons_loss,
                "perceptual_loss": perceptual_loss,
            }
        )
        del outputs["patches"]
        if not return_recons_only:
            del outputs["recon_pixel_values"]
        if not return_action_token_ids_only:
            del outputs["z_rep"]
            del outputs["z_mu"]
            del outputs["z_var"]
        return outputs


    def get_state_dict_to_save(self):
        modules_to_exclude = []
        state_dict = {k: v for k, v in self.state_dict().items() if
                      not any(module_name in k for module_name in modules_to_exclude)}
        return state_dict
