import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

import imageio.v3 as io
import scipy
from scipy import ndimage

from kornia.filters import sobel, Canny

# import open3d as o3d
# from open3d.visualization import draw_geometries, draw



def imshow(img, title="", cmap='gray', cbar=True):
    plt.figure()
    plt.imshow(torch2np(img), cmap=cmap)
    plt.axis('off')
    plt.title(title)
    if (True == cbar):
        plt.colorbar()
    plt.show()

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def torch2np(x):
    if (isinstance(x, torch.Tensor)):
        return x.detach().cpu().numpy().squeeze()
    return x

def torch_sample(img, U, V):
    U = U / (img.shape[1] - 1) * 2 - 1
    V = V / (img.shape[0] - 1) * 2 - 1
    U = U.to(img.dtype)
    V = V.to(img.dtype)
    return F.grid_sample(img.unsqueeze(0).unsqueeze(0),
                         torch.stack([U, V], dim=-1).unsqueeze(0),
                         align_corners=True, mode='nearest',
                         padding_mode='zeros')


# define a function for obtaining soft thresholded indices
def get_uv_soft_thresholding(img, epsilon, temperature=1e-3):
    soft_mask = torch.sigmoid((img - epsilon) / temperature)

    # Generate row and column indices in the same shape as img
    rows, cols = img.shape
    row_indices = torch.arange(rows, device=img.device).float().view(-1, 1).expand(-1, cols)
    col_indices = torch.arange(cols, device=img.device).float().view(1, -1).expand(rows, -1)

    # Apply soft masking to produce UV arrays that approximate each non-zero position
    # Each row in UV will represent an approximate (u, v) position of a non-zero entry
    U_positions = (row_indices * soft_mask).flatten()
    V_positions = (col_indices * soft_mask).flatten()

    # Filter out nearly-zero weights to get approximate non-zero UV positions
    soft_uv = torch.stack([U_positions, V_positions], dim=1)
    non_zero_positions = soft_uv[soft_mask.flatten() > 0.5]

    return non_zero_positions


# variable_floats: 2D numpy array of shape (N, 2) containing row and col coordinates
# Set of (U, V)s in the image: remember the order of the coordinates
# update_value: scalar value to be added to the array
# height, width: height and width of the 2D array
# init_value: initial value of the array
# Returns: updated 2D array with update_value added at the specified coordinates
def soft_indexing_2d(variable_floats, height, width, update_value=1.0, init_value=0):
    # Shape parameters
    rows, cols = height, width

    update_value = torch.tensor([update_value], dtype=variable_floats.dtype, device=variable_floats.device)

    # Generate 2D grid for rows and columns, on the same device as variable_floats
    row_indices = torch.arange(rows).float().unsqueeze(1).expand(-1, cols).to(variable_floats.device)
    col_indices = torch.arange(cols).float().unsqueeze(0).expand(rows, -1).to(variable_floats.device)

    # Initialize a mask array with zeros
    array = torch.zeros((rows, cols), dtype=variable_floats.dtype, device=variable_floats.device) + init_value
    array_updated = array.clone()

    for idx in range(variable_floats.shape[0]):
        # Extract row and col coordinates from variable_floats
        row_float, col_float = variable_floats[idx]

        # Calculate weights for soft indexing in both dimensions
        temperature = 0.1
        row_weights = torch.softmax(-((row_indices - row_float) ** 2)/temperature, dim=0)
        col_weights = torch.softmax(-((col_indices - col_float) ** 2)/temperature, dim=1)

        # Create 2D weights by outer product of row and col weights
        weight_2d = row_weights * col_weights

        # Apply the weighted update
        array_updated += (update_value - array) * weight_2d

    # Assumption: init_value is the minimum value of the array
    array_updated = array_updated.clamp(0 ,1)

    return array_updated


# Using the soft indexing function above, implement the image creation
def get_soft_indexed_image(model_output, height, width):
    # The coordinates in model_output are linearized
    # get the i and j coordinates from that
    model_output_coords = model_output[model_output>0].flatten()
    # out = (h * i + j)/(h*w), get i and j from out
    i = (model_output_coords * (height * width)) / width
    j = (model_output_coords * (height * width)) % width

    # Stack U and V to create a 2D array of shape (N, 2)
    variable_floats = torch.stack([i, j], dim=-1)

    # Call the soft indexing function to get the image
    image = soft_indexing_2d(variable_floats, height, width, update_value=Z)

    return image



def sobel_edge_detection(image):
    if isinstance(image, np.ndarray): # numpy version
        # Define Sobel kernels
        sobel_x = np.array([[-1, 0, 1], 
                            [-2, 0, 2], 
                            [-1, 0, 1]])
        
        sobel_y = np.array([[-1, -2, -1], 
                            [0, 0, 0], 
                            [1, 2, 1]])
        
        # Apply Sobel kernels to get gradients in x and y directions
        grad_x = ndimage.convolve(image, sobel_x)
        grad_y = ndimage.convolve(image, sobel_y)
        
        # Compute the magnitude of the gradient
        edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize the result to range [0, 255]
        edge_magnitude = (edge_magnitude / edge_magnitude.max()) * 255
        return edge_magnitude.astype(np.uint8)

    else: # torch version
        # Define Sobel kernels for x and y gradients
        sobel_x_kernel = torch.tensor([[1, 0, -1],
                                    [2, 0, -2],
                                    [1, 0, -1]], dtype=image.dtype, device=image.device).view(1, 1, 3, 3)
        
        sobel_y_kernel = torch.tensor([[1, 2, 1],
                                    [0, 0, 0],
                                    [-1, -2, -1]], dtype=image.dtype, device=image.device).view(1, 1, 3, 3)

        # Apply Sobel kernels to compute gradients
        # grad_x = F.conv2d(image.unsqueeze(0).unsqueeze(0), sobel_x_kernel, padding=1).squeeze()
        # grad_y = F.conv2d(image.unsqueeze(0).unsqueeze(0), sobel_y_kernel, padding=1).squeeze()
        grad_x = F.conv2d(F.pad(image.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect'), sobel_x_kernel).squeeze()
        grad_y = F.conv2d(F.pad(image.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect'), sobel_y_kernel).squeeze()

        
        # Compute edge magnitude
        edge_magnitude = torch.sqrt(grad_x**2 + grad_y**2) * 255
        return edge_magnitude

# Given an array (list) Z, 
# put it in the image at the locations specified by U, V
# depth_image_torch (used for size): H x W tensor
def get_event_image(U, V, Z, depth_image):
    depth = depth_image.squeeze()
    h, w = depth.shape[0], depth.shape[1]
    U = U.long()
    V = V.long()
    U = U.clamp(0, depth.shape[1] - 1)
    V = V.clamp(0, depth.shape[0] - 1)
    depth_image_Z = torch.zeros([h, w], device=Z.device, dtype=Z.dtype)
    depth_image_Z[V, U] = Z
    return depth_image_Z


def save_point_cloud(U, V, Z, K, filename):
    X = (U - K[0,2]) * Z / K[0,0]
    Y = (V - K[1,2]) * Z / K[1,1]
    points = torch.stack([X, Y, Z], dim=-1)

    points_vis = torch2np(points).reshape(-1, 3)
    points_vis[np.isnan(points_vis)] = 0
    print(f"PC shape: {points_vis.shape}, min: {points_vis.min()}, max: {points_vis.max()}")
    scipy.io.savemat(filename, mdict={'points': points_vis})

# Given two tensors a_pred and b_GT, each of shape (N, 3)
# where each row is a 3D point (x, y, z)
# return a differentiable distance metric between
# the two sets of points
def asymmetric_distance(a_pred, b_GT):
    # print(a_pred.shape, b_GT.shape)
    diff = torch.abs(a_pred.unsqueeze(1) - b_GT) ** 2
    diff = torch.sum(diff, axis=-1)
    diff = torch.sqrt(diff)

    diff_a = torch.sum(torch.amin(diff, axis=1))/a_pred.shape[0]
    diff_b = torch.sum(torch.amin(diff, axis=0))/b_GT.shape[0]

    return diff_a, diff_b

# Given GT of format [H, W, num_bins],
# and pred of the following format:
# event_bin_0: image of size [H, W]
# event_bin_i's: list of all [U1, V1, Z1, T1]
def edge_pipeline_loss(events_bins_GT_np, events_pred):
    events_bins_GT = torch.tensor(events_bins_GT_np.squeeze()).double().to(events_pred["event_bin_0"].device)
    loss = 0
    mse_loss_fn = nn.MSELoss()
    loss_1 = torch.sqrt(mse_loss_fn(events_bins_GT[:, :, 0], events_pred["event_bin_0"]))
    # loss += loss_1
    # print(f"Loss 1: {loss_1}")

    event_bins_pred = events_pred["event_bins"]
    for i in range(events_bins_GT.shape[-1] - 1):
        U1 = event_bins_pred[i]["U1"]
        V1 = event_bins_pred[i]["V1"]
        Z1 = event_bins_pred[i]["Z1"] 
        T1 = event_bins_pred[i]["T1"]
        P1 = torch.stack([U1, V1, T1], dim=-1)

        img_GT = (events_bins_GT[:, :, i+1]>0).float()
        U1_GT, V1_GT = torch.meshgrid(torch.arange(img_GT.shape[0], device=img_GT.device), torch.arange(img_GT.shape[1], device=img_GT.device))
        U1_GT = U1_GT / (img_GT.shape[1] - 1)
        V1_GT = V1_GT / (img_GT.shape[0] - 1)
        T1_GT = img_GT[img_GT > 0]
        U1_GT = U1_GT[img_GT > 0]
        V1_GT = V1_GT[img_GT > 0]
        P1_GT = torch.stack([U1_GT, V1_GT, T1_GT], dim=-1)

        # loss_2_a, loss_2_b = asymmetric_distance(P1, P1_GT)
        # print(f"Loss 2a: {loss_2_a}")
        # print(f"Loss 2b: {loss_2_b}")
        # loss_2 = loss_2_a + loss_2_b
        # loss += loss_2

        UV1 = torch.stack([V1, U1], dim=-1)
        T1_value = 1.0
        h, w = events_bins_GT_np.squeeze().shape[0], events_bins_GT_np.squeeze().shape[1]
        init_value = 0

        # print(f"UV1: {UV1.shape}")
        soft_out = soft_indexing_2d(UV1, h, w, T1_value, init_value)
        # Concatenate soft_out and img_GT for display
        img_both = torch.concatenate([soft_out.squeeze(), img_GT.squeeze()], dim=0)
        imshow(img_both, title="Soft indexing 2D", cbar=True)
        # plt.figure()
        # plt.plot(torch2np(U1), torch2np(V1), 'x')
        # plt.show()
        soft_indexing_2d_loss = ((soft_out.flatten() - img_GT.flatten())**2).mean()

        loss += soft_indexing_2d_loss

    return loss

# Create an event image from events
# events: N x 4 numpy array
# h, w: height and width of the image
# time_alloc: time allocation for each event
# Current time_alloc is a single value for the whole image
def events_to_image(events, h, w, time_alloc):
    image = np.zeros((h, w))
    pol = events[:,3]
    x = events[:,1]
    y = events[:,2]
    neg_p = pol.min()
    image[np.int64(y[pol == neg_p]), np.int64(x[pol == neg_p])] = time_alloc * (np.ones_like(x))[pol == neg_p]
    image[np.int64(y[pol == 1]), np.int64(x[pol == 1])]         = time_alloc * (np.ones_like(x))[pol == 1]

    return image

# Create a bin of events from events
# num_divisions is the number of bins,
# each bin is an image and has a fixed time duration
# events: N x 4 numpy array
# h, w: height and width of the image
def events_to_bins(events, h, w, num_divisions):
    t = (events[:, 0] - events[0, 0])/(events[-1, 0] - events[0, 0])
    x = events[:, 1]
    y = events[:, 2]
    p = events[:, 3]
    time_bins = np.linspace(0, t[-1], num_divisions + 1)
    # get indices with respect to time bins
    frame_events = []
    # Use np.searchsorted to find the starting indices for each frame time
    # left  : a[i-1] < v <= a[i]
    # right : a[i-1] <= v < a[i]
    start_indices = np.searchsorted(t, time_bins[:-1], side='right')
    end_indices = np.searchsorted(t, time_bins[1:], side='left')
    for i, (start, end) in enumerate(zip(start_indices, end_indices)):
        t_mean = (time_bins[i] + time_bins[i+1]) / 2
        img_event = events_to_image(events[start:end], h, w, t_mean)
        frame_events.append(img_event)

    # reverse the list to get the correct order
    frame_events = frame_events[::-1]

    return np.stack(frame_events, axis=-1)

    


# Define a single function for the forward model

# Input:
# depth_image: H x W depth image
# K: 3 x 4 camera matrix
# z_shift: scalar value for shifting the depth image


class EdgePipeline(nn.Module):
    def __init__(self,
                 h,
                 w,
                 K,
                 del_T,
                 num_divisions=4,
                 edge_threshold=7,
                 sigmoid_scale=1,
                 mask = None,
                 visualize_3d_points=False):
        self.K = K
        self.edge_threshold = edge_threshold
        self.visualize_3d_points = visualize_3d_points
        self.h = h
        self.w = w
        self.del_T = del_T
        self.num_divisions = num_divisions
        self.mask = mask
        self.sigmoid_scale = sigmoid_scale
        self.use_old_masking = False
        
    def __call__(self, depth_input, v_shift=0.1):
        del_T = self.del_T
        num_shifts = self.num_divisions - 1
        # reshape depth_input to 2D
        depth_input = depth_input.view(self.h, self.w)

        depth = depth_input.squeeze(0).squeeze(0)

        ## Step I. Get depth edge map
        if (self.mask is not None):
            depth_edge = self.mask
        elif (True == self.use_old_masking):
            depth_edge = sobel_edge_detection(depth)
            # imshow(depth_edge, title="Depth edge image", cbar=True)
            sigmoid_scale = self.sigmoid_scale
            depth_edge = torch.sigmoid(sigmoid_scale * (depth_edge - self.edge_threshold))
        else:
            magnitude, edges = Canny()(depth.unsqueeze(0).unsqueeze(0))
            depth_edge = magnitude
            magnitude[magnitude < self.edge_threshold] = 0
            magnitude[magnitude >= self.edge_threshold] = 1
            depth_edge = magnitude.squeeze()

        self.Z_orig = depth * depth_edge

        ## Step II. Go from depth map to 3D points
        # Z = depth * depth_edge
        Z = self.Z_orig[self.Z_orig > 0]

        U, V = torch.meshgrid(torch.arange(depth.shape[1], device=depth.device), torch.arange(depth.shape[0], device=depth.device), indexing="xy")
        U = U.float()
        V = V.float()
        U = U[self.Z_orig > 0]
        V = V[self.Z_orig > 0]

        X = (U - self.K[0, 2]) * Z / self.K[0, 0]
        Y = (V - self.K[1, 2]) * Z / self.K[1, 1]

        points = torch.stack([X, Y, Z], dim=-1)
        points = points.reshape(-1, 3)

        z_shift = v_shift * del_T

        event_bins = []
        # Do this num_shifts number of times
        for iter in range(num_shifts):
            zshift = (iter + 1) * z_shift
            ## Step III. Go from 3D points to shifted 3D points
            zero = torch.tensor([0.0]).to(depth.device)
            shift = torch.stack([zero, zero, zshift]).T
            points_Z1 = points + shift

            # Step IV. Project 3D points to 2D
            # points_Z1 = points_Z1.view(depth.shape[0], depth.shape[1], 3)

            X1 = points_Z1[..., 0]
            Y1 = points_Z1[..., 1]
            Z1 = points_Z1[..., 2]

            U1 = (self.K[0, 0] * X1 / Z1) + self.K[0, 2]
            # U1 = U1.clamp(0, depth.shape[1] - 1)
            V1 = (self.K[1, 1] * Y1 / Z1) + self.K[1, 2]
            # V1 = V1.clamp(0, depth.shape[0] - 1)
            
            # U1 = U1[~torch.isnan(Z)]
            # V1 = V1[~torch.isnan(Z)]
            # Z1 = Z1[~torch.isnan(Z)]

            out_coords = {}
            out_coords["U1"] = U1
            out_coords["V1"] = V1
            out_coords["U1_norm"] = U1 / (depth.shape[1] - 1)
            out_coords["V1_norm"] = V1 / (depth.shape[0] - 1)
            out_coords["T1"] = del_T * (iter + 0.5) * torch.ones_like(U1)
            out_coords["Z1"] = Z1

            event_bins.append(out_coords)

        # U = U[~torch.isnan(Z)]
        # V = V[~torch.isnan(Z)]
        # Z = Z[~torch.isnan(Z)]
        output = {}
        output["U"] = U
        output["V"] = V
        output["Z"] = Z
        output["T"] = del_T * (num_shifts + 0.5)
        output["event_bin_0"] = depth_edge * output["T"]
        output["event_bins"] = event_bins


        if self.visualize_3d_points:
            # NOT used in computational graph
            # Step EXTRA. Make an image from these points
            imshow(depth_edge, title="Depth (binary) edge image", cbar=True)
            imshow(Z, title="Edge depth image", cbar=True)
            imshow(torch.isnan(Z), title="Nan mask", cbar=True)
            imshow(U, title="U", cbar=True)
            imshow(V, title="V", cbar=True)

            imshow(points[..., 2].reshape(depth.shape), title="3D points Z", cbar=True)
            imshow(points[..., 0].reshape(depth.shape), title="3D points X", cbar=True)
            imshow(points[..., 1].reshape(depth.shape), title="3D points Y", cbar=True)
            
            imshow(U1, title="U1", cbar=True)
            imshow(V1, title="V1", cbar=True)

            depth_image_Z1 = get_event_image(U1, V1, Z1, depth)
            depth_image_Z = get_event_image(U, V, Z, depth)
            imshow(depth_image_Z, title="Depth image Zoro", cbar=True)
            imshow(depth_image_Z1, title="Depth image Zoro1", cbar=True)

        return output


    # deprecated function - don't use and maintain this
    def edge_pipeline_np(self, depth, z_shift=0.1):
        K = self.K
        edge_threshold = self.edge_threshold
        visualize_3d_points = self.visualize_3d_points

        # TODO: probably copy is unnecessary since depth variable is not used later
        depth_no_nan = depth


        ## Step I. Get depth edge map
        depth_edge = sobel_edge_detection((1/depth_no_nan))
        depth_edge = (depth_edge>edge_threshold).astype(np.float64)


        ## Step II. Go from depth map to 3D points
        Z = depth_no_nan * depth_edge
        # Make the background as nan, saves lines and computation later
        Z[Z==0] = np.nan

        U, V = np.meshgrid(np.arange(Z.shape[1]), np.arange(Z.shape[0]))
        U = U.astype(np.float32)
        V = V.astype(np.float32)
        U[np.isnan(Z)] = np.nan
        V[np.isnan(Z)] = np.nan
        

        X = (U - K[0,2]) * Z / K[0,0]
        Y = (V - K[1,2]) * Z / K[1,1]


        points = np.stack([X, Y, Z], axis=-1)
        points = points.reshape(-1, 3)
        
        ## Step III. Go from 3D points to shifted 3D points
        points_Z1 = points + np.array([0, 0, z_shift]).astype(np.float64)

        
        # Step IV. Project 3D points to 2D
        points_Z1 = points_Z1.reshape(depth.shape[0], depth.shape[1], 3)


        X1 = points_Z1[..., 0]
        Y1 = points_Z1[..., 1]
        Z1 = points_Z1[..., 2]

        U1 = (K[0,0] * X1 / Z1) + K[0,2]
        U1 = np.clip(U1, 0, depth.shape[1]-1)
        V1 = (K[1,1] * Y1 / Z1) + K[1,2]
        V1 = np.clip(V1, 0, depth.shape[0]-1)

        U1_orig = U1.copy()
        V1_orig = V1.copy()
        U1[np.isnan(U1)] = 0
        V1[np.isnan(V1)] = 0
        U1 = U1.astype(int)
        V1 = V1.astype(int)

        # Step V. Go from shifted 3D points to events
        event_image = np.zeros([depth.shape[0], depth.shape[1], 3]).astype(np.float64)
        U[np.isnan(U)] = 0
        V[np.isnan(V)] = 0
        U = U.astype(int)
        V = V.astype(int)
        event_image[V, U, 0] += 1
        event_image[V1, U1, 2] += 1

        # Step EXTRA. Make an image from these points
        Z1[np.isnan(U1_orig)] = np.nan
        depth_image_Z1 = np.zeros_like(depth) + np.nan
        depth_image_Z1[V1, U1] = Z1



        if visualize_3d_points:
            imshow(depth_edge, title="Depth (binary) edge image", cbar=True)
            imshow(Z, title="Edge depth image", cbar=True)

            imshow(U * (~np.isnan(Z)), title="U", cbar=True)
            imshow(V * (~np.isnan(Z)), title="V", cbar=True)

            imshow(points[..., 2].reshape(depth.shape), title="3D points Z", cbar=True)
            imshow(points[..., 0].reshape(depth.shape), title="3D points X", cbar=True)
            imshow(points[..., 1].reshape(depth.shape), title="3D points Y", cbar=True)

            X_vis = X.copy()
            Y_vis = Y.copy()
            Z_vis = Z.copy()
            X_vis[np.isnan(X_vis)] = 0
            Y_vis[np.isnan(Y_vis)] = 0
            Z_vis[np.isnan(Z_vis)] = 0
            points_vis = np.stack([X_vis, Y_vis, Z_vis], axis=-1)
            points_vis = points_vis.reshape(-1, 3)

            # 3D points visualization for original points
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(points_vis)
            # draw([pcd])
            del X_vis, Y_vis, Z_vis, points_vis


            imshow(points_Z1[..., 2], title="3D points Z1", cbar=True)
            imshow(points_Z1[..., 0], title="3D points X1", cbar=True)
            imshow(points_Z1[..., 1], title="3D points Y1", cbar=True)

            # 3D points visualization for shifted points
            points_Z1_vis = points_Z1.copy().reshape(-1, 3)
            points_Z1_vis[np.isnan(points_Z1_vis)] = 0
            print(points_Z1_vis.shape)
            print(points_Z1_vis.min())
            print(points_Z1_vis.max())

            # pcd1 = o3d.geometry.PointCloud()
            # pcd1.points = o3d.utility.Vector3dVector(points_Z1_vis)
            # draw([pcd1])
            del points_Z1_vis

            imshow(U1_orig, title="U1", cbar=True)
            imshow(V1_orig, title="V1", cbar=True)
            del U1_orig, V1_orig

            imshow(points_Z1[..., 2], title="GT Depth image Z1", cbar=True)
            imshow(depth_image_Z1, title="Depth image Z1 - reprojected", cbar=True)        


        return event_image
