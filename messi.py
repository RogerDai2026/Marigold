#%%

# 7 November 2024 Thursday

import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as io

# Read the GT depth image and analyze it
fname = "/home/ad150/git/Marigold/script/input/Camera_events/encoded_full/PYRAMIDAL/event_frame_0000.tif"
events = io.imread(fname)

plt.figure()
plt.imshow(events)
plt.colorbar()

plt.figure()
plt.imshow(events[:20, :20, :])
plt.colorbar()

plt.figure()
plt.imshow(events[:20, :20, 0])
plt.colorbar()
plt.figure()
plt.imshow(events[:20, :20, 1])
plt.colorbar()
plt.figure()
plt.imshow(events[:20, :20, 2])
plt.colorbar()

#%%

# encode the events using VAE and decode them again


# %%

# 3 November 2024, Sunday

import numpy as np
import matplotlib.pyplot as plt

# Read the GT depth image and analyze it
# fnameGT = "/shared/ad150/event3d/carla/multimodal/Town05/sequence_1_val/depth/data/05_001_0123_depth.npy"
fnameGT = "/shared/ad150/event3d/carla/Town10_val/depth/data/depth_0000000547.npy"
depthGT = np.load(fnameGT)

depthGT_10m = np.clip(depthGT, a_min=0, a_max=10)
depthGT_20m = np.clip(depthGT, a_min=0, a_max=20)
depthGT_30m = np.clip(depthGT, a_min=0, a_max=30)

# fig, ax = plt.subplots(2, 2, figsize=(10, 10))
# ax[0,1].imshow(depthGT_10m)
# ax[0,1].set_title("Depth GT 10m")
# ax[1,0].imshow(depthGT_20m)
# ax[1,0].set_title("Depth GT 20m")
# ax[1,1].imshow(depthGT_30m)
# ax[1,1].set_title("Depth GT 30m")


# Read the predicted depth image and analyze it
# fnamePred = "/home/ad150/git/Marigold/output/eval/carla_test/prediction/multimodal/Town05/sequence_1_val/events/frames_event/LINEAR/pred_event_frame_0123.npy"
# fnamePred = "/home/ad150/git/Marigold/output/eval/carla_test/prediction/Town10_val/events/frames_event/LINEAR/pred_event_frame_0547.npy"
fnamePred = "/home/ad150/git/Marigold/output/eval/carla_test_v1_cosine/prediction/Town10_val/events/frames_event/LINEAR/pred_event_frame_0547.npy"
log_disp_norm = np.load(fnamePred)
# log_disp = log_disp_norm * (norm_max - norm_min) + norm_min
# disp = np.exp(log_disp)
# depthPred = 1/disp
depthPred = log_disp_norm * 250.0

norm_min = np.log(1/(250+1e-6))
norm_max = np.log(1/1e-6)
log_disp = np.log(1/(depthPred+1e-6))
log_disp_norm = (log_disp - norm_min) / (norm_max - norm_min)

depthPred_10m = np.clip(depthPred, a_min=0, a_max=10)
depthPred_20m = np.clip(depthPred, a_min=0, a_max=20)
depthPred_30m = np.clip(depthPred, a_min=0, a_max=30)

# fig, ax = plt.subplots(2, 2, figsize=(10, 10))
# ax[0,0].imshow(depthPred)
# ax[0,0].set_title("Depth Pred")
# ax[0,1].imshow(depthPred_10m)
# ax[0,1].set_title("Depth Pred 10m")
# ax[1,0].imshow(depthPred_20m)
# ax[1,0].set_title("Depth Pred 20m")
# ax[1,1].imshow(depthPred_30m)
# ax[1,1].set_title("Depth Pred 30m")


fig, ax = plt.subplots(4, 2, figsize=(10, 10))
ax[0,0].imshow(depthGT)
ax[0,0].set_title("Depth GT")
ax[0,1].imshow(depthPred)
ax[0,1].set_title("Depth Pred")
ax[1,0].imshow(depthGT_10m)
ax[1,0].set_title("Depth GT 10m")
ax[1,1].imshow(depthPred_10m)
ax[1,1].set_title("Depth Pred 10m")
ax[2,0].imshow(depthGT_20m)
ax[2,0].set_title("Depth GT 20m")
ax[2,1].imshow(depthPred_20m)
ax[2,1].set_title("Depth Pred 20m")
ax[3,0].imshow(depthGT_30m)
ax[3,0].set_title("Depth GT 30m")
ax[3,1].imshow(depthPred_30m)
ax[3,1].set_title("Depth Pred 30m")


#%%




























#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as io
import torch

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def float2image(x):
    # if numpy
    if isinstance(x, np.ndarray):
        return np.uint8(normalize(x)*255)
    # if torch
    if isinstance(x, torch.Tensor):
        return torch.uint8(normalize(x)*255)


#%%

# check consecutive depth frames
# in carla dataset

n = 10

base_dir = "/shared/ad150/event3d/carla/multimodal/Town05/sequence_1_val/"
extension = ".npy"
fluff = "depth/data/05_001_"

fluff_ev = "events/frames_event/LINEAR/event_frame_"
extension_ev = ".tif"

fluff_full_ev = "events/data/05_001_"
extension_full_ev = "_events.npz"

filename1 = os.path.join(base_dir, f"{fluff}{n:04d}_depth" + extension)
filename0 = os.path.join(base_dir, f"{fluff}{n-1:04d}_depth" + extension)
filename1_ev = os.path.join(base_dir, f"{fluff_ev}{n:04d}" + extension_ev)
filename1_full_ev = os.path.join(base_dir, f"{fluff_full_ev}{n:04d}" + extension_full_ev)


depth_n = np.load(filename1)
depth_n_1 = np.load(filename0)
events_n = io.imread(filename1_ev)

events_all = np.load(filename1_full_ev)

#%%

def make_dictionary_from_events(events):
    x = events["x"]
    y = events["y"]
    ts = events["t"]
    p = events["p"]

    events_dict = {}

    for i in range(len(x)):
        if (x[i], y[i]) in events_dict:
            events_dict[(x[i], y[i])].append(ts[i])
        else:
            events_dict[(x[i], y[i])] = [ts[i]]

    return events_dict


def make_events_from_dictionary(events_dict, H, W):
    events = np.zeros((H, W, 3))
    for key in events_dict.keys():
        events[key[1], key[0], 0] = len(events_dict[key])
        events[key[1], key[0], 1] = np.mean(events_dict[key])
        events[key[1], key[0], 2] = np.std(events_dict[key])
    return events

events_dict = make_dictionary_from_events(events_all)
events_n_all = make_events_from_dictionary(events_dict, depth_n.shape[0], depth_n.shape[1])

num_events = events_n_all[:,:,0]
mean_events = events_n_all[:,:,1]
std_events = events_n_all[:,:,2]

plt.figure()
plt.imshow(num_events)
plt.title("Number of events")
plt.colorbar()

plt.figure()
plt.imshow(mean_events)
plt.title("Mean of timestamps")
plt.colorbar()

plt.figure()
plt.imshow(std_events)
plt.title("Std of timestamps")
plt.colorbar()

#%%


mask = depth_n < 250

# plt.figure()
# plt.imshow(depth_n * mask, cmap="magma")
# plt.colorbar()

# plt.figure()
# plt.imshow(depth_n_1 * mask, cmap="magma")
# plt.colorbar()

plt.figure()
plt.imshow(np.abs(depth_n - depth_n_1)<1, cmap="magma")
plt.colorbar()

plt.figure()
plt.imshow(np.log(np.abs(depth_n - depth_n_1)+1)>0.1, cmap="magma")
plt.colorbar()

plt.figure()
plt.imshow(np.sum(events_n, axis=-1)>0)
plt.colorbar()

#%%

# binarize difference using sigmoid

diff_sigmoid = torch.sigmoid(torch.tensor(np.abs(depth_n - depth_n_1)))
diff_sigmoid = diff_sigmoid.numpy()

plt.figure()
plt.imshow(diff_sigmoid)
plt.colorbar()

plt.figure()
plt.imshow(diff_sigmoid * (diff_sigmoid>0.6))
plt.colorbar()

plt.figure()
plt.imshow(np.sum(events_n, axis=-1)>0)
plt.colorbar()


#%%

disparity_n = 1/depth_n
disparity_n_1 = 1/depth_n_1
disparity_diff = np.clip(np.log(np.abs(disparity_n - disparity_n_1)),a_min=-15, a_max=0)
disparity_diff_temp = np.abs(disparity_n - disparity_n_1)

event_img = np.sum(events_n, axis=-1)>0

plt.figure()
# plt.imshow(disparity_diff * (disparity_diff > -5) - 5 * (disparity_diff <= -5))
# plt.imshow(torch.sigmoid(torch.tensor(disparity_diff_temp)).numpy())
plt.imshow(np.abs(disparity_n))
plt.colorbar()

plt.figure()
plt.imshow(event_img)
plt.colorbar()

plt.figure()
plt.imshow(np.abs(disparity_diff * event_img))
plt.colorbar()




#%%
event_img = np.sum(events_n, axis=-1)>0

threshold = 0.001
disparity_diff = np.log(np.abs(1/depth_n - 1/depth_n_1)+1)
mask = disparity_diff > threshold
disparity_diff_masked = disparity_diff * mask

threshold_depth = 1
depth_diff = np.abs(depth_n - depth_n_1)
depth_mask = depth_diff < threshold
depth_diff_masked = depth_diff * depth_mask

# Plot them both together in subplots
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(np.sign(disparity_diff_masked), cmap="magma")
ax[0].set_title(f"Disparity diff (threshold: {threshold})")
ax[0].set_axis_off()
ax[1].imshow(np.sign(depth_diff_masked), cmap="magma")
ax[1].set_title(f"Depth diff (threshold: {threshold_depth})")
ax[1].set_axis_off()
ax[2].imshow(np.sum(events_n, axis=-1)>0)
ax[2].set_title("Events")
ax[2].set_axis_off()
plt.show()

# %%

# Function that will count the number of events 
# in a frame around a window for each pixel

def count_events_around_window(events, window_size=3):
    # Get the shape of the events
    H = events.shape[0]
    W = events.shape[1]
    # Initialize the count array
    count = np.zeros((H, W))
    # Iterate over the pixels
    for i in range(H):
        for j in range(W):
            # Get the window around the pixel
            i_min = max(i - window_size, 0)
            i_max = min(i + window_size + 1, H)
            j_min = max(j - window_size, 0)
            j_max = min(j + window_size + 1, W)
            # Count the number of events in the window
            try:
                event_subimg = np.sign(np.sum(events[i_min:i_max, j_min:j_max, :], axis=-1))
            except:
                event_subimg = events[i_min:i_max, j_min:j_max]
            # sum of the total number of events in the window
            count[i, j] = np.sum(event_subimg)
    return count

def std_of_timestamps_around_window(events, window_size=3):
    # Get the shape of the events
    H = events.shape[0]
    W = events.shape[1]
    # Initialize the std_img array
    std_img = np.zeros((H, W))
    # Iterate over the pixels
    for i in range(H):
        for j in range(W):
            # Get the window around the pixel
            i_min = max(i - window_size, 0)
            i_max = min(i + window_size + 1, H)
            j_min = max(j - window_size, 0)
            j_max = min(j + window_size + 1, W)
            # Gather the events in the window
            event_subimg = events[i_min:i_max, j_min:j_max, :2]
            # Take only non zero events in first channel
            event_subimg_1 = event_subimg[:,:,0][(event_subimg[:,:,0] != 0)]
            # Take only non zero events in second channel
            event_subimg_2 = event_subimg[:,:,1][(event_subimg[:,:,1] != 0)]

            event_subimg = np.append(event_subimg_1, event_subimg_2)
            
            if (len(event_subimg) > 0):
                # std of the events in the window
                std_img[i, j] = np.std(event_subimg_1)

    return std_img

events_count = count_events_around_window(events_n, window_size=1)
events_std = std_of_timestamps_around_window(events_n, window_size=3)

# Plot them both together in subplots
fig, ax = plt.subplots(2, 2, figsize=(15, 5))
ax[0,0].imshow(events_n)
ax[0,0].set_title(f"Events")
ax[0,0].set_axis_off()
# ax[0,1].imshow(np.log(events_count+1), cmap="magma")
ax[0,1].imshow(events_count)
ax[0,1].set_title(f"Events count")
ax[0,1].set_axis_off()
ax[1,0].imshow((np.log(1/depth_n)*event_img*-1), cmap="magma")
ax[1,0].set_title("Disparity diff")
ax[1,0].set_axis_off()
ax[1,1].imshow(events_std) #, cmap="magma")
ax[1,1].set_title("standard deviation in window")
ax[1,1].set_axis_off()
plt.show()

#%%

## SOMETHING GOOD

import matplotlib.pyplot as plt

from skimage import data
from skimage import exposure
from skimage.exposure import match_histograms, equalize_hist

# image = normalize(events_count*event_img)
# reference = normalize(1/depth_n*event_img)
# vmax = 0.1
image = float2image(events_count*event_img)
reference = float2image(1/depth_n*event_img)
vmax = 25

# matched = match_histograms(image, reference, channel_axis=-1)

# image_eq = (equalize_hist(image))
# reference_eq = (equalize_hist(reference))
image_eq = float2image(equalize_hist(image))
reference_eq = float2image(equalize_hist(reference))


fig, ax = plt.subplots(
    nrows=3, ncols=2, figsize=(8, 3), sharex=True, sharey=True
)
for aa in ax.flatten():
    aa.set_axis_off()

ax[0,0].imshow(image)
ax[0,0].set_title('Source')
ax[1,0].imshow(image_eq)
ax[1,0].set_title('Source eq')

ax[0,1].imshow(reference)
ax[0,1].set_title('Reference')
ax[1,1].imshow(reference_eq)
ax[1,1].set_title('Reference eq')

# take difference, convert to float before subtracting
diff_vanilla = np.abs(np.float32(image) - np.float32(reference))
diff_eq = np.abs(np.float32(image_eq) - np.float32(reference_eq))


ax[2,0].imshow(diff_vanilla, vmax=vmax)
ax[2,0].set_title('Source - Reference')
ax[2,1].imshow(diff_eq, vmax=vmax)
ax[2,1].set_title('Source eq - Reference eq')

plt.tight_layout()
plt.show()

#%%

# pytorch equivalent of box filter

import torch

def box_filter(x, r):
    # x: (B, C, H, W)
    # r: scalar
    B, C, H, W = x.shape
    print(x.shape)
    # write a code using conv2d with a kernel of size 2r+1

    conv = torch.nn.Conv2d(C, C, kernel_size=2*r+1, padding=r, bias=False)
    print(conv.weight.data.shape)
    conv.weight.data = torch.ones_like(conv.weight.data)

    return conv(x)

events_nT = torch.tensor(np.float32(events_n), requires_grad=True)
events_nT_binary = torch.sign(torch.sum(events_nT, axis=-1))

x = events_nT_binary.unsqueeze(0).unsqueeze(0).float()
# events_n is of shape (H, W, C), so make it (B, C, H, W)
r = 1
events_count_T = box_filter(x, r)

plt.figure()
plt.imshow(x.squeeze(0).permute(1, 2, 0).squeeze(0).cpu().detach().numpy())
plt.colorbar()

events_count_TN = (np.sum(events_count_T.squeeze(0).permute(1, 2, 0).squeeze(0).cpu().detach().numpy(), axis=-1))

plt.figure()
plt.imshow(events_count_TN)
plt.colorbar()

#%%

# torchvision based histogram equalization

depth_n_T = torch.tensor(np.float32(depth_n), requires_grad=True)
event_img_T = torch.tensor(np.float32(event_img))

image = events_count_T*event_img_T
reference = 1/depth_n_T*event_img_T

#%%

from torchvision.transforms import functional as F

image1 = np.uint8(image*255)
reference1 = np.uint8(reference*255)

imageT = torch.tensor(image1).unsqueeze(0).unsqueeze(0).float().requires_grad_(True)
referenceT = torch.tensor(reference1).unsqueeze(0).unsqueeze(0)

image_eqT = F.equalize(imageT)
reference_eqT = F.equalize(referenceT)

imageTN = imageT.squeeze(0).squeeze(0).cpu().numpy()
referenceTN = referenceT.squeeze(0).squeeze(0).cpu().numpy()
image_eqTN = image_eqT.squeeze(0).squeeze(0).cpu().numpy()
reference_eqTN = reference_eqT.squeeze(0).squeeze(0).cpu().numpy()


fig, ax = plt.subplots(
    nrows=3, ncols=2, figsize=(8, 3), sharex=True, sharey=True
)
for aa in ax.flatten():
    aa.set_axis_off()

ax[0,0].imshow(imageTN)
ax[0,0].set_title('Source')
ax[1,0].imshow(image_eqTN)
ax[1,0].set_title('Source eq')

ax[0,1].imshow(referenceTN)
ax[0,1].set_title('Reference')
ax[1,1].imshow(reference_eqTN)
ax[1,1].set_title('Reference eq')

ax[2,0].imshow(np.abs(imageTN - referenceTN), vmax=255)
ax[2,0].set_title('Source - Reference')
ax[2,1].imshow(np.abs(image_eqTN - reference_eqTN), vmax=255)
ax[2,1].set_title('Source eq - Reference eq')

plt.tight_layout()
plt.show()


#%% 

# pytorch basded histogram matching

from pytorch_histogram_matching import Histogram_Matching
import torch

HM = Histogram_Matching(differentiable=True)

imageT = torch.tensor(image).unsqueeze(0).unsqueeze(0)
referenceT = torch.tensor(reference).unsqueeze(0).unsqueeze(0)

matchedT = HM(imageT, referenceT)


fig, (ax1, ax2, ax3) = plt.subplots(
    nrows=1, ncols=3, figsize=(8, 3), sharex=True, sharey=True
)
for aa in (ax1, ax2, ax3):
    aa.set_axis_off()

ax1.imshow(imageT.squeeze(0).squeeze(0).cpu().numpy())
ax1.set_title('Source')
ax2.imshow(referenceT.squeeze(0).squeeze(0).cpu().numpy())
ax2.set_title('Reference')
ax3.imshow(matchedT.squeeze(0).squeeze(0).cpu().numpy())
ax3.set_title('Matched')

plt.tight_layout()
plt.show()

#%%

#%%

bce_loss_fn = torch.nn.BCELoss()

loss = bce_loss_fn(torch.tensor(events_count), torch.tensor(1/depth_n*event_img))



#%%


# Display the events in DENSE paper style

depth = np.load("/shared/ad150/event3d/carla/Town10_val/depth/data/depth_0000000547.npy")

disp = 1/(depth) # * (depth<1000))
disp_log = np.log(disp)
disp_log_masked = np.clip(disp_log, a_min = -4.5, a_max = 5)

plt.figure()
plt.imshow(disp_log_masked, cmap="magma")
plt.colorbar()


depth = np.load("/home/ad150/git/Marigold/script/output/events/depth_npy/event_frame_0547_pred.npy")

disp = 1/(depth) # * (depth<1000))
disp_log = np.log(disp)
disp_log_masked = np.clip(disp_log, a_min = None, a_max = 5)

plt.figure()
plt.imshow(disp_log_masked, cmap="magma")
plt.colorbar()

#%%

import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure

a = torch.zeros([50,50])
b = torch.zeros([50,50])
b[10:12, 10:12] = 1

ap = a.unsqueeze(0).unsqueeze(0)
bp = b.unsqueeze(0).unsqueeze(0)

ssim = StructuralSimilarityIndexMeasure()

print(ssim(ap, bp))






#%%

import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as io

depth1 = np.load("/shared/ad150/event3d/carla/Town10_val/depth/data/depth_0000000547.npy")




#%%

# Analyze generated images

import numpy as np
import matplotlib.pyplot as plt

depth_full = np.load("/home/ad150/git/Marigold/script/output/events/train_marigold_from_marigold/depth_npy/event_frame_0000_full_pred.npy")
depth_high = np.load("/home/ad150/git/Marigold/script/output/events/train_marigold_from_marigold/depth_npy/event_frame_0000_high_pred.npy")
depth_low = np.load("/home/ad150/git/Marigold/script/output/events/train_marigold_from_marigold/depth_npy/event_frame_0000_low_pred.npy")

depth_GT = np.load("/shared/ad150/event3d/carla/multimodal/Town05/sequence_1_val/depth/data/05_001_0010_depth.npy")
depth_GT = np.clip(depth_GT, a_min=0, a_max=250)

depth_full = torch.tensor(depth_full, requires_grad=True)
depth_high = torch.tensor(depth_high, requires_grad=True)
depth_low = torch.tensor(depth_low, requires_grad=True)

diff_for_events = torch.sigmoid(torch.abs(depth_full - depth_low))

event_diff_img = io.imread("/home/ad150/git/Marigold/script/input/events/event_frame_0000_high.tif")
event_diff_img = np.sum(event_diff_img, axis=-1)>0

plt.figure()
plt.imshow(((diff_for_events)).detach().cpu().numpy())
plt.title("diff_for_events")
plt.colorbar()

plt.figure()
plt.imshow(event_diff_img)
plt.title("events")
plt.colorbar()

#%%

bce_loss_fn = torch.nn.BCELoss()
bce_loss = bce_loss_fn(torch.sigmoid(diff_for_events), torch.tensor(event_diff_img).float())
print(bce_loss)
bce_loss = bce_loss_fn(torch.tensor(event_diff_img).float()*torch.sigmoid(diff_for_events), torch.tensor(event_diff_img).float())
print(bce_loss)
bce_loss = bce_loss_fn(depth_high, depth_low)
print(bce_loss)
bce_loss = bce_loss_fn(depth_high, depth_high)
print(bce_loss)
print("")

l1_loss_fn = torch.nn.L1Loss()
l1_loss = l1_loss_fn(torch.sigmoid(diff_for_events), torch.tensor(event_diff_img).float())
print(l1_loss)
l1_loss = l1_loss_fn(torch.tensor(event_diff_img).float()*torch.sigmoid(diff_for_events), torch.tensor(event_diff_img).float())
print(l1_loss)
l1_loss = l1_loss_fn(depth_full, depth_low)
print(l1_loss)
l1_loss = l1_loss_fn(depth_full, depth_high)
print(l1_loss)
print("")

# %%
