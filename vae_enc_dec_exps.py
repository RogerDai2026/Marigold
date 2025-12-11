#%%

os.environ["CUDA_VISIBLE_DEVICES"]="1"

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision

import imageio.v3 as io
import matplotlib.pyplot as plt

import os

from marigold.marigold_pipeline import MarigoldPipeline


base_ckpt_dir = "/shared/ad150/event3d/marigold/checkpoint/marigold-v1-0-pretrained"
pretrained_path = "stable-diffusion-2"

_pipeline_kwargs = {}

# Load the pretrained model
model = MarigoldPipeline.from_pretrained(
        os.path.join(base_ckpt_dir, pretrained_path), **_pipeline_kwargs
    ).to("cuda")

print(model.vae)
print(f"Model loaded from {os.path.join(base_ckpt_dir, pretrained_path)}")


#%%

def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
    """
    Encode RGB image into latent.

    Args:
        rgb_in (`torch.Tensor`):
            Input RGB image to be encoded.

    Returns:
        `torch.Tensor`: Image latent.
    """
    # encode
    h = self.vae.encoder(rgb_in)
    moments = self.vae.quant_conv(h)
    mean, logvar = torch.chunk(moments, 2, dim=1)
    # scale latent
    rgb_latent = mean * self.rgb_latent_scale_factor
    return rgb_latent

def decode_rgb(self, rgb_latent: torch.Tensor) -> torch.Tensor:
    """
    Decode image latent into RGB image.

    Args:
        rgb_latent (`torch.Tensor`):
            Image latent to be decoded.

    Returns:
        `torch.Tensor`: Decoded RGB image.
    """
    # scale latent
    rgb_latent = rgb_latent / self.rgb_latent_scale_factor
    # decode
    z = self.vae.post_quant_conv(rgb_latent)
    rgb = self.vae.decoder(z)
    return rgb

#%%

# fname = "sobel_edge_example.png"
# fname = "/shared/ad150/event3d/carla/Town10_val/events/frames_event/PYRAMIDAL/event_frame_0143.tif"
# fname = "/home/ad150/git/Marigold/script/input/Camera_events/encoded_full/PYRAMIDAL/event_frame_0000.tif"
fname = "/home/ad150/git/Marigold/script/input/Camera_events/encoded_full/VAE_ROBUST/event_frame_0000.tif"
# rgb_in = io.imread(fname)[:,:,:3] # [h, w, c]
rgb_in = io.imread(fname)[:80, :80, :] # [h, w, c]
# Convert to tensor of shape [1, c, h, w] and normalize
rgb_in_tensor = torch.tensor(rgb_in).permute(2, 0, 1).unsqueeze(0).float() / 255.0
rgb_in_tensor = rgb_in_tensor.to(model.device)
rgb_in_tensor = rgb_in_tensor * 2 - 1

plt.figure()
plt.imshow(rgb_in)
plt.axis("off")
plt.title("Input RGB")
# plt.colorbar()

rgb_latent = encode_rgb(model, rgb_in_tensor)
print(rgb_latent.shape)

rgb_out_tensor = decode_rgb(model, rgb_latent)
print(rgb_out_tensor.shape)

# Convert to numpy array
rgb_out_np = rgb_out_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
rgb_out_np = np.uint8(255 * (rgb_out_np+1)/2)

plt.figure()
plt.imshow(rgb_out_np)
plt.title("Encoded -> Decoded RGB")
plt.axis("off")
# plt.colorbar()

# For curiosity, also plot the latent vector
plt.figure()
plt.imshow(rgb_latent.squeeze().permute(1, 2, 0).detach().cpu().numpy())
plt.title("Latent")
plt.axis("off")
# plt.colorbar()

#%%
loss = F.mse_loss(rgb_out_tensor, rgb_in_tensor)
print(f"Loss: {loss}")



# %%
a = np.random.randint(0, 2, [100,100]).astype(np.float32)
plt.figure()
plt.imshow(a)
plt.colorbar()

from scipy.ndimage import gaussian_filter
blurred = gaussian_filter(a, sigma=0.75)

plt.figure()
plt.imshow(blurred)
plt.colorbar()






#%%

fname = "/shared/ad150/event3d/carla/Town10_val/depth/data/depth_0000000547.npy"
depth_in = np.load(fname)

plt.figure()
plt.imshow(depth_in)
plt.colorbar()

depth_in = (depth_in - depth_in.min()) / (depth_in.max() - depth_in.min())
depth_in = depth_in * 2 - 1
depth_in = np.repeat(depth_in[:, :, np.newaxis], 3, axis=2)

depth_in_tensor = torch.tensor(depth_in).permute(2, 0, 1).unsqueeze(0).float()
depth_in_tensor = depth_in_tensor.to(model.device)

depth_latent = encode_rgb(model, depth_in_tensor)
print(depth_latent.shape)

depth_out_tensor = decode_rgb(model ,depth_latent)
print(depth_out_tensor.shape)

#%%

depth_out_np = depth_out_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
# depth_out_np = (depth_out_np - depth_out_np.min()) / (depth_out_np.max() - depth_out_np.min())
depth_out_np = (depth_out_np + 1) / 2
depth_out_np = np.clip(depth_out_np, 0, 1)

depth_out_np = depth_out_np * 1000
log_disp = np.log((depth_out_np+1))
log_disp_norm = (log_disp - log_disp.min()) / (log_disp.max() - log_disp.min())

plt.figure()
plt.imshow(log_disp_norm)
plt.colorbar()
