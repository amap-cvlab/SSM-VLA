import torch.nn.functional as F
import math
import cv2
from PIL import Image, ImageFont, ImageDraw
import os
import torchvision.transforms as T
import numpy as np

def visualize_latent_motion_reconstruction(
    initial_frame,
    next_frame,
    recons_next_frame,
    latent_motion_ids,
    path
):  
    initial_frame = initial_frame[0]
    c, h, w = initial_frame.shape
    h = h + 30
    initial_frame = T.ToPILImage()(initial_frame)
    pred_frame = next_frame.shape[0]
    compare_img = Image.new('RGB', size=((1+2*pred_frame)*w, h))
    draw_compare_img = ImageDraw.Draw(compare_img)
    compare_img.paste(initial_frame, box=(0, 0))
    next_frame_backup = next_frame.clone()
    recons_next_frame_backup = recons_next_frame.clone()
    for i in range(pred_frame):
        next_frame = T.ToPILImage()(next_frame_backup[i])
        recons_next_frame = T.ToPILImage()(recons_next_frame_backup[i])
        compare_img.paste(next_frame, box=((2*i+1)*w, 0))
        compare_img.paste(recons_next_frame, box=((2*i+2)*w, 0))
    
    latent_motion_ids = latent_motion_ids.numpy().tolist()
    font_path = os.path.join(cv2.__path__[0],'qt','fonts','DejaVuSans.ttf')
    font = ImageFont.truetype(font_path, size=12)
    draw_compare_img.text((w, h-20), f"{latent_motion_ids}", font=font, fill=(0, 255, 0))
    compare_img.save(path)

def visualize_latent_motion_depth_reconstruction(
    initial_frame, # b, 1, h, w
    next_frame,
    recons_next_frame,
    latent_motion_ids,
    path
):
    # 修复 initial_frame 处理
    initial_frame_i = initial_frame[0].cpu().numpy()
    # 确保维度正确 - 完全压缩所有冗余维度
    initial_frame_i = initial_frame_i.squeeze()
    # 如果压缩后不是2D，则尝试获取正确的2D表示
    if initial_frame_i.ndim > 2:
        initial_frame_i = initial_frame_i.reshape(-1, initial_frame_i.shape[-1])
    
    initial_frame_i = (initial_frame_i - initial_frame_i.min()) / (initial_frame_i.max() - initial_frame_i.min() + 1e-8) * 255.0
    initial_frame_i = np.stack([initial_frame_i, initial_frame_i, initial_frame_i], axis=-1).astype(np.uint8)
    initial_frame_img = Image.fromarray(initial_frame_i)
    
    # 获取图像尺寸
    h, w = initial_frame_i.shape[:2]
    h = h + 30
    
    pred_frame = next_frame.shape[0]
    compare_img = Image.new('RGB', size=((1+2*pred_frame)*w, h))
    draw_compare_img = ImageDraw.Draw(compare_img)
    compare_img.paste(initial_frame_img, box=(0, 0))
    next_frame_backup = next_frame.clone()
    recons_next_frame_backup = recons_next_frame.clone()
    for i in range(pred_frame):
        # 修复 next_frame 处理
        next_frame_i = next_frame_backup[i].cpu().numpy()
        # 确保维度正确 - 完全压缩所有冗余维度
        next_frame_i = next_frame_i.squeeze()
        # 如果压缩后不是2D，则尝试获取正确的2D表示
        if next_frame_i.ndim > 2:
            next_frame_i = next_frame_i.reshape(-1, next_frame_i.shape[-1])
        
        next_frame_i = (next_frame_i - next_frame_i.min()) / (next_frame_i.max() - next_frame_i.min() + 1e-8) * 255.0
        next_frame_i = np.stack([next_frame_i, next_frame_i, next_frame_i], axis=-1).astype(np.uint8)
        next_frame_img = Image.fromarray(next_frame_i)
        
        # 修复 recons_next_frame 处理
        recons_frame_i = recons_next_frame_backup[i].cpu().numpy()
        # 确保维度正确
        recons_frame_i = recons_frame_i.squeeze()
        # 如果压缩后不是2D，则尝试获取正确的2D表示
        if recons_frame_i.ndim > 2:
            recons_frame_i = recons_frame_i.reshape(-1, recons_frame_i.shape[-1])
        
        recons_frame_i = (recons_frame_i - recons_frame_i.min()) / (recons_frame_i.max() - recons_frame_i.min() + 1e-8) * 255.0
        recons_frame_i = np.stack([recons_frame_i, recons_frame_i, recons_frame_i], axis=-1).astype(np.uint8)
        recons_frame_img = Image.fromarray(recons_frame_i)

        compare_img.paste(next_frame_img, box=((2*i+1)*w, 0))
        compare_img.paste(recons_frame_img, box=((2*i+2)*w, 0))
    
    latent_motion_ids = latent_motion_ids.numpy().tolist()
    font_path = os.path.join(cv2.__path__[0],'qt','fonts','DejaVuSans.ttf')
    font = ImageFont.truetype(font_path, size=12)
    draw_compare_img.text((w, h-20), f"{latent_motion_ids}", font=font, fill=(0, 255, 0))
    compare_img.save(path)