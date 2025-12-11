#%%
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import torch
from torchviz import make_dot

from tqdm.autonotebook import tqdm

from edge_pipeline import *
from siren import *

import wandb

device = torch.device("cuda:3")

# Set a fixed seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# # write the autoreload magic command
# %load_ext autoreload
# %autoreload 2

# Load depth image
fname = "data/depth_0000000008.npy"
# fname = "data/depth_0000000011.npy"
depth_image = np.load(fname)
amax = 100.0
amin = 1.0
depth_image[depth_image < amin] = float('nan')
depth_image[depth_image > amax] = float('nan')
depth_image[np.isnan(depth_image)] = amin
# depth_image = np.clip(depth_image, amin, amax)
h_orig = depth_image.shape[0]
w_orig = depth_image.shape[1]

crop = 100
crop_end = -100
depth_image = depth_image[crop:crop_end, crop:crop_end]

viz = True
if (True == viz):
    imshow(depth_image, title="Depth image")

h = depth_image.shape[0]
w = depth_image.shape[1]

depth_image_torch = torch.from_numpy(depth_image).unsqueeze(0).unsqueeze(0).to(device)

# Read event in numpy format
event_np = np.load("data/events_0000000008.npy")

s = 1e9
ms = 1e6
us = 1e3
# del_T = 10 * ms
num_divisions = 4
# time interval bewteen min and max is normalized to 1
del_T = (1- 0) / num_divisions

event_bins_GT = events_to_bins(event_np, h_orig, w_orig, num_divisions)

event_bins_GT = event_bins_GT[crop:crop_end, crop:crop_end, :]

# if (True == viz):
#     for i in range(num_divisions):
#         imshow(event_bins_GT[:, :, i], title=f"Event bin {i}, min, max = {event_bins_GT[:, :, i].min(), event_bins_GT[:, :, i].max()}", cbar=True)


# Load camera matrix
# K  = np.array([[194.8389461655774, 0.0,               170.20896993269332, -19.635286970460974],
#                [0.0,               194.8389461655774, 127.00404928416845, 0.0],
#                [0.0,               0.0,               1.0,                0.0]]) 
K  = np.array([[194.8389461655774, 0.0,               70.20896993269332, -19.635286970460974],
               [0.0,               194.8389461655774, 27.00404928416845, 0.0],
               [0.0,               0.0,               1.0,                0.0]]) 
# Set z_shift
edge_threshold=0.5
visualize_3d_points = False

edge_pipeline_simulation = EdgePipeline(h,
                                        w,
                                        K,
                                        del_T=del_T,
                                        num_divisions=num_divisions,
                                        edge_threshold=edge_threshold,
                                        sigmoid_scale=1e10,
                                        visualize_3d_points=visualize_3d_points)


################################################
# ### DATA LOADING ###
# # Read the actual event image
# event_fname = "data/frame_0000000008.png"
# # event_fname = "data/event_frame_00011.tif"
# event_image_gt = io.imread(event_fname)
# imshow(np.sum(event_image_gt, axis=-1) > 0, title="GT event image", cbar=True)
# # imshow(event_image_gt, title="GT event image", cbar=True)



################################################

### Test edge_pipeline_simulation

depth_image_torch.requires_grad = True
VS = 10.0
v_shift = torch.tensor([VS]).float().to(device).requires_grad_(True)
print(v_shift)

event_output = edge_pipeline_simulation(depth_image_torch, v_shift)

Ugt = event_output["U"]
Vgt = event_output["V"]
Zgt = event_output["Z"]
Tgt = event_output["T"]
event_bin_0 = event_output["event_bin_0"]
event_bins = event_output["event_bins"]


print(f"requires grad: U: {Ugt.requires_grad}, V: {Vgt.requires_grad}, Z: {Zgt.requires_grad}")
depth_image_Z_gt = get_event_image(Ugt, Vgt, Zgt, depth_image_torch)
i = num_divisions - 1
if (True == viz):
    # make_dot(U1gt)
    imshow(depth_image_Z_gt, title=f"Depth image Zoro", cmap="magma", cbar=True) #, min, max = {depth_image_Z.min(), depth_image_Z.max()}", cbar=True)

for i in range(1, num_divisions):
    U1gt = event_bins[i-1]["U1"]
    V1gt = event_bins[i-1]["V1"]
    Z1gt = event_bins[i-1]["Z1"]
    T1gt = event_bins[i-1]["T1"]
    print(f"requires grad: U1: {U1gt.requires_grad}, V1: {V1gt.requires_grad}, Z1: {Z1gt.requires_grad}")
    depth_image_Z1_gt = get_event_image(U1gt, V1gt, Z1gt, depth_image_torch)
    if (True == viz):
        imshow(depth_image_Z1_gt, title=f"Event image Zoro{i}", cmap="magma", cbar=True) #, min, max = {event_image_Z1.min(), event_image_Z1.max()}", cbar=True)
    # imshow(event_bins_GT[:, :, i], title=f"Event bin {i}", cbar=True)


save_point_cloud(Ugt, Vgt, Zgt, K, "points_Z_gt.mat")

#%% ################################################
### Test loss function
# loss = edge_pipeline_loss(event_bins_GT, event_output)

# print(f"Loss: {loss}")
# print(f"loss.requires_grad: {loss.requires_grad}")
# loss.backward()

#%% ################################################
### EDGE dataloader

v_shift_simulation = torch.tensor([VS]).float()
depth_image_np = depth_image_torch.detach().cpu().numpy()
event_bins_data = EDGEData(depth_image_np, event_bins_GT, edge_pipeline_simulation, v_shift_simulation)
dataloader = DataLoader(event_bins_data, batch_size=1, pin_memory=True, num_workers=0)

edge_siren = Siren(in_features=2, out_features=1, hidden_features=256, 
                      hidden_layers=3, outermost_linear=True)
edge_siren.to(device)





#%% ################################################

### Fit SIREN to the edge image

h = depth_image.squeeze().shape[0]
w = depth_image.squeeze().shape[1]

## Step II. Test SIREN on depth image

optim = torch.optim.Adam(lr=1e-4, params=edge_siren.parameters())

model_input, gt = next(iter(dataloader))
model_input = model_input.to(device)
gt = {key: value.to(device) for key, value in gt.items()}



real_data = False
if (True == real_data):
    gt_type = "event_bins_GT"
else:
    gt_type = "event_bins_simulated"


mask = gt[gt_type].squeeze()[:,:,0] > 0
mask = torch2np(mask.float())
mask = mask * 2.0 - 1.0
mask_torch = torch.from_numpy(mask).to(device)

# define loss_fn
loss_fn = nn.MSELoss()
















#%%

# ###### MODEL PREDICTS COORDINATES WHERE THERE ARE EDGES ######

# total_steps = 10000
# steps_til_summary = 500

# optim = torch.optim.Adam(lr=1e-3, params=edge_siren.parameters())

# # Add scheduler
# # lr_scheduler will decrease the learning rate
# # by a factor of gammma every step_size steps
# scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=total_steps//8, gamma=0.5)

# # define tqmd progress bar
# pbar = tqdm(range(total_steps))

# loss_mse_all = []

# ### Optimization loop ###
# for step in pbar:
#     start_time = time.time()

#     model_output, coords = edge_siren(model_input)

#     # Soft_indexing + mse loss
#     model_output_image = get_soft_indexed_image(model_output.view(h, w), h, w)

#     loss_mse = ((model_output_image.flatten() - mask_torch.flatten())**2).mean()
#     loss_mse_all.append(loss_mse.item())


#     # train_loss = loss_mse
#     train_loss = loss_mse

#     pbar.set_description(f"Step {step}, Total loss {train_loss}, iteration time {time.time() - start_time}")
#     pbar.refresh()


#     if not step % steps_til_summary:
#         print(f"model_output: {model_output.shape}, coords: {coords.shape}")
#         print("Step %d, Total loss %0.6f, iteration time %0.6f" % (step, train_loss, time.time() - start_time))
#         print(f"Step {step}, Total loss {train_loss}, iteration time {time.time() - start_time}, current learning rate: {scheduler.get_last_lr()}")

#         fig, axes = plt.subplots(1, 2, figsize=(15, 6))

#         # Prediced Row
#         axes[0].imshow(torch2np(model_output.squeeze().view(h, w)))
#         axes[0].set_title(f"Predicted image")
#         axes[1].imshow(torch2np(mask_torch))
#         axes[1].set_title(f"Mask GT")
#         plt.show()


#     optim.zero_grad()
#     train_loss.backward()
#     optim.step()
#     scheduler.step()

# #%%

# plt.figure()
# plt.plot(loss_mse_all, label="Total loss")
# plt.legend()
# plt.show()


###### MODEL PREDICTS COORDINATES WHERE THERE ARE EDGES ######














































#%%


total_steps = 500
steps_til_summary = 100

optim = torch.optim.Adam(lr=1e-4, params=edge_siren.parameters())

# Add scheduler
# lr_scheduler will decrease the learning rate
# by a factor of gammma every step_size steps
# scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=total_steps//32, gamma=0.5)

# define tqmd progress bar
pbar = tqdm(range(total_steps))

loss_mse_all = []
loss_a_all = []
loss_b_all = []
loss_c_all = []
loss_d_all = []
loss_chamfer1_all = []
loss_chamfer2_all = []
loss_chamferv0_all = []
loss_soft_indexing_all = []

mask_fit = torch.from_numpy(depth_image_np).to(device)
mask_fit = mask_fit/amax * 2.0 - 1.0

### Optimization loop ###
for step in pbar:
    start_time = time.time()

    model_output, coords = edge_siren(model_input)

    # MSE Loss
    loss_mse = ((model_output.flatten() - mask_fit.flatten())**2).mean()
    loss_mse_all.append(loss_mse.item())
    
    # Asymmmetric loss
    # losses_ad = edge_mse_test(model_output.view(h, w), mask_torch.view(h, w))
    # loss_a_all.append(losses_ad["loss_a"].item())
    # loss_b_all.append(losses_ad["loss_b"].item())
    # # loss_c_all.append(losses_ad["loss_c"].item())
    # # loss_d_all.append(losses_ad["loss_d"].item())

    # # Chamfer distance
    # loss_chamfer1, loss_chamfer2 = chamfer_distance_loss(model_output.view(h, w), mask_torch.view(h, w))
    # loss_chamfer1_all.append(loss_chamfer1.item())
    # loss_chamfer2_all.append(loss_chamfer2.item())
    # # loss_chamfer = loss_chamfer1 + loss_chamfer2

    # # Chamfer distance v0
    # loss_chamferv0 = chamfer_distance_loss_v0(model_output.view(h, w), mask_torch.view(h, w))
    # loss_chamferv0_all.append(loss_chamferv0.item())

    # # Soft_indexing + mse loss
    # loss_soft_indexing = soft_indexing_loss_v0(model_output.view(h, w), mask_torch.view(h, w))
    # loss_soft_indexing_all.append(loss_soft_indexing.item())


    # train_loss = loss_mse
    train_loss = loss_mse

    pbar.set_description(f"Step {step}, Total loss {train_loss}, iteration time {time.time() - start_time}")
    pbar.refresh()

    if not step % steps_til_summary:
        print(f"model_output: {model_output.shape}, coords: {coords.shape}")
        print("Step %d, Total loss %0.6f, iteration time %0.6f" % (step, train_loss, time.time() - start_time))
        # print(f"Step {step}, Total loss {train_loss}, iteration time {time.time() - start_time}")
        print(f"Step {step}, Total loss {train_loss}, iteration time {time.time() - start_time}")#, current learning rate: {scheduler.get_last_lr()}")

        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Prediced Row
        axes[0].imshow(torch2np(model_output.squeeze().view(h, w)))
        axes[0].set_title(f"Predicted image")
        axes[1].imshow(torch2np(mask_fit))
        axes[1].set_title(f"Mask GT")
        plt.show()


    optim.zero_grad()
    train_loss.backward()
    optim.step()
    # scheduler.step()

#%%

plt.figure()
plt.plot(loss_mse_all, label="Total loss")
plt.plot(loss_a_all, label="Loss A")
plt.plot(loss_b_all, label="Loss B")
plt.plot(loss_chamfer1_all, label="Chamfer loss 1")
plt.plot(loss_chamfer2_all, label="Chamfer loss 2")
plt.legend()

#%% ################################################




























#%%
# ### Test EDGE dataloader
# model_input, gt = next(iter(dataloader))

# e_GT = gt["event_bins_GT"].squeeze()
# e_sim = gt["event_bins_simulated"].squeeze()

# for i in range(num_divisions):
#     imshow(e_GT[:, :, i], title=f"Event bin {i}, min, max = {e_GT[:, :, i].min(), e_GT[:, :, i].max()}", cbar=True)
# for i in range(num_divisions):
#     imshow(e_sim[:, :, i], title=f"Event bin {i}, min, max = {e_sim[:, :, i].min(), e_sim[:, :, i].max()}", cbar=True)

# get current date and time in string format
import datetime
now = datetime.datetime.now()
now_str = now.strftime("%Y-%m-%d_%H-%M-%S")
wandb_run_name = f"edge_inr_{now_str}"

# Initialize wandb
wandb.init(project="EDGE", name=wandb_run_name)


#%% ################################################

# make this true to write results locally
write_results = False
# make this true to visualize the results
viz = True

# make this true if running on real data
real_data = False
if (True == real_data):
    gt_type = "event_bins_GT"
    mask = event_bins_GT.squeeze()[:,:,0] > 0
    mask = torch2np(mask)
else:
    gt_type = "event_bins_simulated"
    mask = depth_image_Z_gt > 0
    mask = torch2np(mask)
    
mask_torch = torch.from_numpy(mask).to(device)
### Optimization loop ###

edge_pipeline = EdgePipeline(h,
                             w,
                             K,
                             del_T=del_T,
                             num_divisions=num_divisions,
                             edge_threshold=edge_threshold,
                             mask = None,
                             visualize_3d_points=visualize_3d_points)
                            #  mask = torch.ones_like(mask_torch),

h = depth_image.squeeze().shape[0]
w = depth_image.squeeze().shape[1]

v_shift =  torch.tensor([VS]).float().to(device).requires_grad_(True) #torch.tensor([2.0/del_T]).to(device).requires_grad_(True)
print(v_shift)

## Step II. Test SIREN on depth image
total_steps = 10
steps_til_summary = 1

optim = torch.optim.Adam(lr=5e-5, params=edge_siren.parameters())
# optim = torch.optim.Adam(lr=1e-4, params=[{'params': edge_siren.parameters()},
#                                           {'params': v_shift}])

loss_all = []

# Add scheduler
# lr_scheduler will decrease the learning rate
# by a factor of gammma every step_size steps
# scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=total_steps//8, gamma=0.5)

model_input, gt = next(iter(dataloader))
gt = {key: value.to(device) for key, value in gt.items()}
model_input = model_input.to(device)
pbar = tqdm(range(total_steps))

for step in pbar:
    start_time = time.time()

    model_output, coords = edge_siren(model_input)
    model_output = torch.clamp(model_output, -1, 1)
    # model_output_transformed = ((model_output + 1)/2 * amax).flatten() * mask_torch.flatten()
    model_output_transformed = ((model_output + 1)/2 * amax)
    train_loss = edge_mse_v2(model_output_transformed, gt[gt_type], edge_pipeline, v_shift)
    pbar.set_description(f"Step {step}, Total loss {train_loss}, iteration time {time.time() - start_time}")
    pbar.refresh()

    loss_all.append(train_loss.item())
    try:
        wandb.log({"loss": train_loss.item()})
    except:
        pass

    if not step % steps_til_summary:
        print(f"model_output: {model_output.shape}, coords: {coords.shape}, v_shift: {v_shift}")
        print("Step %d, Total loss %0.6f, iteration time %0.6f" % (step, train_loss, time.time() - start_time))

        depth_pred = model_output.cpu().view(h, w).detach().numpy()
        
        out_event_fwd = edge_pipeline(model_output_transformed, v_shift)
        pred_event_bin_0 = out_event_fwd["event_bin_0"].cpu().view(h, w).detach().numpy()

        event_bins = out_event_fwd["event_bins"]
        U1 = event_bins[i-1]["U1"]
        V1 = event_bins[i-1]["V1"]
        Z1 = event_bins[i-1]["Z1"]
        T1 = event_bins[i-1]["T1"]
        bin_1 = get_event_image(U1, V1, T1, depth_image_torch)
        pred_event_bin_1 = bin_1.cpu().view(h, w).detach().numpy()
        del out_event_fwd

        gt_event_bins = gt[gt_type].squeeze()
        disp_event_bin_0 = gt_event_bins[:, :, 0].view(h, w).detach().cpu().numpy()
        disp_event_bin_1 = gt_event_bins[:, :, i].view(h, w).detach().cpu().numpy()
        
        event_image_Z1 = get_event_image(U1, V1, Z1, model_output_transformed.view(h, w))


        if ((True == real_data) and (True == viz)):
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            # Prediced Row
            axes[0,0].imshow(depth_pred, cmap="magma")
            axes[0,0].set_title(f"Predicted depth image")
            axes[0,1].imshow(torch2np(model_output_transformed.view(h, w) * mask_torch))
            axes[0,1].set_title(f"Predicted depth image masked")
            axes[0,2].imshow(torch2np(event_image_Z1))
            axes[0,2].set_title(f"Predicted depth image Zoro{i}")

            # GT Row
            axes[1,0].imshow(depth_image_np.squeeze(), cmap="magma")
            axes[1,0].set_title(f"GT depth image for reference")
            axes[1,1].imshow(torch2np(disp_event_bin_0))
            axes[1,1].set_title(f"GT event image - bin 0")
            axes[1,2].imshow(torch2np(disp_event_bin_1))
            axes[1,2].set_title(f"GT event image - bin {i}")
            plt.show()
        elif ((False == real_data) and (True == viz)):
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            # Predicted Row
            axes[0,0].imshow(depth_pred, cmap="magma")
            axes[0,0].set_title(f"Predicted depth image")
            axes[0,1].imshow(torch2np(model_output_transformed.view(h, w) * mask_torch), cmap="magma")
            axes[0,1].set_title(f"Predicted depth image masked")
            axes[0,2].imshow(torch2np(event_image_Z1), cmap="magma")
            axes[0,2].set_title(f"Predicted depth image Zoro{i}")

            # GT Row
            axes[1,0].imshow(depth_image_np.squeeze(), cmap="magma")
            axes[1,0].set_title(f"GT depth image")
            axes[1,1].imshow(torch2np(depth_image_Z_gt), cmap="magma")
            axes[1,1].set_title(f"GT depth image - bin 0")
            axes[1,2].imshow(torch2np(depth_image_Z1_gt), cmap="magma")
            axes[1,2].set_title(f"GT depth image - bin {i}")
            plt.show()

        if (True == write_results):
            os.makedirs(f"results/{wandb_run_name}", exist_ok=True)
            d1 = depth_pred * amax
            d1_edge = torch2np(model_output_transformed.view(h, w) * mask_torch)
            d1_edge = d1_edge.astype(np.uint8)
            e_i = torch2np(event_image_Z1).astype(np.uint8)

            # Concatenate each of them with GT
            d1_edge = np.concatenate([d1_edge, torch2np(depth_image_Z_gt).astype(np.uint8)], axis=1)
            e_i = np.concatenate([e_i, torch2np(depth_image_Z1_gt).astype(np.uint8)], axis=1)

            io.imwrite(f"results/{wandb_run_name}/depth_pred_edge_{step}.png", d1_edge)
            io.imwrite(f"results/{wandb_run_name}/depth_pred_edge_Zoro{i}_{step}.png", e_i)
            
            # save_point_cloud(U1, V1, Z1, K, f"points_Z_{step}.mat")


    optim.zero_grad()
    train_loss.backward()
    optim.step()
    # scheduler.step()

#%%

plt.figure()
plt.plot(loss_all, label="Total loss")
plt.legend()
plt.show()

imshow(model_output_transformed.view(h, w), title="Predicted depth image", cbar=True)

#%%

# imshow(model_output_transformed.view(h, w) * mask_torch, title="Predicted depth image masked", cbar=True)
event_image_Z1 = get_event_image(U1, V1, Z1, depth_image)
imshow(event_image_Z1, title=f"Event image Zoro{i}", cbar=True)
imshow(event_image_Z1, title=f"Event image Zoro{i}", cbar=True)

#%%
# # 3D points visualization for shifted points


# # X1 = (U1 - K[0,2]) * Z1 / K[0,0]
# # Y1 = (V1 - K[1,2]) * Z1 / K[1,1]
# # points_Z1 = torch.stack([X1, Y1, Z1], dim=-1)

# # points_Z1_vis = torch2np(points_Z1).reshape(-1, 3)
# # points_Z1_vis[np.isnan(points_Z1_vis)] = 0
# # print(points_Z1_vis.shape)
# # print(points_Z1_vis.min())
# # print(points_Z1_vis.max())

# # # save the points as a matlab file
# # import scipy.io
# # scipy.io.savemat('points_Z1.mat', mdict={'points_Z1': points_Z1_vis})
# # also save GT points


# #%%

# plt.figure()
# plt.imshow(np.zeros((h, w)))
# plt.plot(torch2np(U1), torch2np(V1), 'x')
# plt.show()


# #%% ################ numpy version ################
# # edge_pipeline.visualize_3d_points = True
# # event_fwd_np = edge_pipeline.edge_pipeline_np(depth_image)

# # # imshow(np.sum(event_fwd_np, axis=-1) > 0, title="Simulated event image", cbar=True)
# # imshow(event_fwd_np, title="Simulated event image", cbar=True)

# # # Read the actual event image
# # event_fname = "data/frame_0000000008.png"
# # # event_fname = "data/event_frame_00011.tif"
# # event_image_gt = io.imread(event_fname)
# # imshow(np.sum(event_image_gt, axis=-1) > 0, title="GT event image", cbar=True)
# # # imshow(event_image_gt, title="GT event image", cbar=True)
# ####################################################

# #%%

# # Some test code

# # a_pred = torch.tensor([[1.1, 2.1, 3.1],
# #                        [1.1, 2.1, 3.1],
# #                        [1.1, 2.1, 3.1],
# #                        [1.1, 2.1, 3.1]]).requires_grad_(True)
# # b_GT = torch.tensor([[1.0, 2.0, 3.0],
# #                      [1.0, 2.0, 3.0]])

# # Create a random tensor of floats with size (4, 3)
# a_pred = torch.rand(32567, 3, requires_grad=True)
# print(a_pred)
# b_GT = torch.rand(1560, 3)
# print(b_GT)

# loss1, loss2 = asymmetric_distance(a_pred, b_GT)

# print(f"loss1: {loss1}")
# print(f"loss2: {loss2}")


# #%%

# from chamferdist import ChamferDistance

# source_cloud = torch.rand(1, 250*350, 3).requires_grad_(True).cuda()
# target_cloud = torch.rand(1, 250*350, 3).requires_grad_(True).cuda()

# print(source_cloud.requires_grad)
# print(target_cloud.requires_grad)

# chamferDist = ChamferDistance()
# dist_forward = chamferDist(source_cloud, target_cloud)
# dist_backward = chamferDist(target_cloud, source_cloud)
# print(dist_forward)
# print(dist_forward.requires_grad)
# print(dist_backward)
# print(dist_backward.requires_grad)
# # %%






# #%%

# # Some test code
# import torch

# import torch

# # Define the float variable with requires_grad=True
# variable_float = torch.tensor([2.5], requires_grad=True)

# # Array to update
# array1 = torch.zeros(10)

# # Soft indexing setup
# length = array1.size(0)
# weights = torch.softmax(-((torch.arange(length).float() - variable_float) ** 2), dim=0)

# # The value you want to "place" at the index represented by variable_float
# update_value = torch.tensor(5.0)

# # Perform the soft update
# array1_updated = array1 + weights * (update_value - array1)

# print("Updated array1:", array1_updated)



# #%%

# # Some test code
# import torch

# height = 250
# width = 350

# # Define the 2D float array of "indices" (each entry is [row, col])
# # variable_float = torch.tensor([[25.1, 24.9], [75.1, 74.9]], requires_grad=True)
# variable_float = torch.rand([625, 2]) * torch.tensor([height, width])
# variable_float.requires_grad = True



# # Perform the soft indexing
# array1_updated = soft_indexing_2d(variable_float, height=height, width=width, update_value=5.0, init_value=0)

# imshow(torch.zeros_like(array1_updated))
# imshow(array1_updated)

# # Visualize the computation graph using make_dot
# # from torchviz import make_dot
# # make_dot(array1_updated)


# #%%

# import torch

# # Initialize img with requires_grad=True and some non-zero values
# img = torch.tensor([[0.0, 1.0, 0.0],
#                     [1.0, -0.1, 0.9],
#                     [0.6, 0.8, -0.9]], requires_grad=True)

# # Set a threshold to identify non-zero values softly (using a small epsilon)
# epsilon = 1e-3
# soft_mask = torch.sigmoid((img - epsilon) * 10) - 0.5

# # Generate UV positions with a differentiable weighted approach
# rows, cols = img.shape
# row_indices = torch.arange(rows).float().unsqueeze(1).expand(-1, cols)
# col_indices = torch.arange(cols).float().unsqueeze(0).expand(rows, -1)

# # Get soft positions weighted by the non-zero approximation
# U = (row_indices * soft_mask).sum() / soft_mask.sum()
# V = (col_indices * soft_mask).sum() / soft_mask.sum()

# U1 = row_indices * soft_mask
# V1 = col_indices * soft_mask

# print(U1)
# print(V1)

# mask = (U1>0) * (V1>0)
# print(mask)

# U_final = U1[mask]
# V_final = V1[mask]
# print(U_final)
# print(V_final)



# # The positions (U, V) will be differentiable approximations of non-zero positions
# print("Soft UV positions:", U, V)






# #%%

# # Some test code
# import torch

# # Define img with requires_grad=True and some non-zero values
# # set torch random seeed
# torch.manual_seed(0)
# epsilon = 0.5
# # img = torch.tensor([[0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
# #                     [2.0, 0.0, 3.0, 2.0, 0.0, 3.0],
# #                     [0.0, 4.0, 0.0, -0.6, 4.0, 0.0],
# #                     [0.0, 1.0, 0.0, -0.9, 1.0, 0.0],
# #                     [2.0, 0.0, 3.0, 2.0, 0.0, -0.75],
# #                     [0.0, 4.0, 0.0, 0.0, 4.0, 0.0]], requires_grad=True)
# img = torch.rand(50,50).requires_grad_(True)
# imshow(img)
# imshow(img > epsilon)

# # Define a threshold and soft mask for non-zero values
# temperature = 0.001

# # Get the soft thresholded indices
# non_zero_positions = get_uv_soft_thresholding(img, epsilon, temperature)


# #%%

# height = img.shape[0]
# width = img.shape[1]

# # plot the non_zero_positions
# img_non_zero = soft_indexing_2d(non_zero_positions, height=height, width=width, update_value=1.0, init_value=0)
# imshow(img_non_zero.clamp(0, 1))

# #%%

# plt.figure()
# plt.plot(torch2np(non_zero_positions[:, 1]), torch2np(non_zero_positions[:, 0]), 'x')
# plt.show()


#%%

# from kornia.filters import sobel, Canny

# depth_image[np.isnan(depth_image)] = 100
# disp_image = np.log(1/(depth_image+1e-6))

# input = torch.from_numpy(depth_image).unsqueeze(0).unsqueeze(0).requires_grad_(True)

# magnitude, edges = Canny()(input)  # 5x3x4x4

# magnitude[magnitude < 0.5] = 0
# magnitude[magnitude >= 0.5] = 1

# imshow(magnitude, title="Canny Magnitude", cbar=False)

# make_dot(magnitude)

# #%%
# depth_sobel = sobel(input)

# imshow(depth_sobel, title="Depth sobel")
# print(depth_sobel.requires_grad)


# depth_sobel[depth_sobel < 2.0 * depth_sobel.mean()] = 0
# print(depth_sobel.requires_grad)
# imshow(depth_sobel)

# depth_sobel = torch.sigmoid(10*(depth_sobel-0.1))
# imshow(depth_sobel, "binarized Sobel")
# print(depth_sobel.requires_grad)

# #%%

# from kornia.filters import in_range

# lower = torch.tensor([0.5]).unsqueeze(0).unsqueeze(0).unsqueeze(0)
# upper = torch.tensor([1.0]).unsqueeze(0).unsqueeze(0).unsqueeze(0)
# mask = in_range(depth_sobel, lower, upper, return_mask=True)
# imshow(depth_sobel)
# imshow(mask)
# print(mask.requires_grad)
# %%
