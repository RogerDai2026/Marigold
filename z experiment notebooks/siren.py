import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
from typing import OrderedDict

from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
import matplotlib.pyplot as plt
import scipy.ndimage

import time

from edge_pipeline import EdgePipeline, get_event_image, edge_pipeline_loss
from edge_pipeline import asymmetric_distance, imshow, soft_indexing_2d, get_uv_soft_thresholding

from chamferdist import ChamferDistance


def get_mgrid_square(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

def get_mgrid(shape, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    if (isinstance(shape, int)):
        return get_mgrid_square(shape, dim)
    tensors = tuple(torch.linspace(-1, 1, steps=shape[i]) for i in range(dim))
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

def get_cameraman_tensor(sidelength):
    img = Image.fromarray(skimage.data.camera())        
    transform = Compose([
        Resize(sidelength),
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    img = transform(img)
    return img

def gradients_mse(model_output, coords, gt_gradients):
    # compute gradients on the model
    gradients = gradient(model_output, coords)
    # compare them with the ground-truth
    gradients_loss = torch.mean((gradients - gt_gradients).pow(2).sum(-1))
    return gradients_loss


def edge_mse(model_output,
            gt_edge,
            edge_pipeline : EdgePipeline,
            z_shift):
    # compute gradients on the model
    event_fwd_torch = edge_pipeline(model_output, z_shift)
    # TODO: Change this function to something better
    # Currently it is used to convert from 3 channels to 1 channel just by
    # taking the max value of the 3 channels
    event_fwd_torch = torch.amax(event_fwd_torch, axis=-1).view(-1, 1)

    # compare them with the ground-truth
    edge_loss = torch.mean((event_fwd_torch - gt_edge).pow(2))
    return edge_loss

def edge_mse_v2(model_output,
                event_bins_gt,
                edge_pipeline : EdgePipeline,
                v_shift):
    # obtain events_pred from model_output
    events_pred = edge_pipeline(model_output, v_shift)
    return edge_pipeline_loss(event_bins_gt, events_pred)

# model_output: torch.tensor of shape (H, W)
# mask: torch.tensor of shape (H, W)
def edge_mse_test(model_output, mask):
    model_output_transformed = (model_output - model_output.min()) / (model_output.max() - model_output.min())
    edge_indices = model_output_transformed > 0.5
    if (edge_indices.sum() == 0):
        print("No edge points found")
    model_output_edge_points = model_output_transformed[edge_indices]
    U1, V1 = torch.meshgrid(torch.arange(model_output.shape[0], device=model_output.device), torch.arange(model_output.shape[1], device=model_output.device))
    U1 = U1[edge_indices]
    V1 = V1[edge_indices]
    U1 = U1 / (model_output.shape[1] - 1)
    V1 = V1 / (model_output.shape[0] - 1)
    T1 = model_output_edge_points * torch.ones_like(U1)
    P1 = torch.stack([U1, V1, T1], dim=-1)

    U1_GT, V1_GT = torch.meshgrid(torch.arange(model_output.shape[0], device=model_output.device), torch.arange(model_output.shape[1], device=model_output.device))
    U1_GT = U1_GT[mask > 0.5]
    V1_GT = V1_GT[mask > 0.5]
    U1_GT = U1_GT / (model_output.shape[1] - 1)
    V1_GT = V1_GT / (model_output.shape[0] - 1)
    T1_GT = torch.ones_like(U1_GT)
    P1_GT = torch.stack([U1_GT, V1_GT, T1_GT], dim=-1)

    loss_2_a, loss_2_b = asymmetric_distance(P1, P1_GT)
    # loss_2_a, loss_2_b = asymmetric_distance(P1, P1_GT)
    # print(f"loss_2_a: {loss_2_a}, loss_2_b: {loss_2_b}")

    # # # Do the same for the background points
    # background_indices = model_output_transformed <= 0.5
    # U1b, V1b = torch.meshgrid(torch.arange(model_output.shape[0], device=model_output.device), torch.arange(model_output.shape[1], device=model_output.device))
    # U1b = U1b[background_indices]
    # V1b = V1b[background_indices]
    # U1b = U1b / (model_output.shape[1] - 1)
    # V1b = V1b / (model_output.shape[0] - 1)
    # T1b = model_output_transformed[background_indices]
    # P1b = torch.stack([U1b, V1b, T1b], dim=-1)

    # U1b_GT, V1b_GT = torch.meshgrid(torch.arange(model_output.shape[0], device=model_output.device), torch.arange(model_output.shape[1], device=model_output.device))
    # U1b_GT = U1b_GT[mask <= 0.5]
    # V1b_GT = V1b_GT[mask <= 0.5]
    # U1b_GT = U1b_GT / (model_output.shape[1] - 1)
    # V1b_GT = V1b_GT / (model_output.shape[0] - 1)
    # T1b_GT = torch.zeros_like(U1b_GT)
    # P1b_GT = torch.stack([U1b_GT, V1b_GT, T1b_GT], dim=-1)

    # loss_3_a, loss_3_b = asymmetric_distance(P1b, P1b_GT)
    # print(f"loss_3_a: {loss_3_a}, loss_3_b: {loss_3_b}")

    # model_output_background_points = model_output_transformed[model_output <= 0.5]
    # loss_4 = torch.mean(model_output_background_points.pow(2))
    # print(f"loss_4: {loss_4}")

    losses = {}
    losses["loss_a"] = loss_2_a
    losses["loss_b"] = loss_2_b
    # losses["loss_c"] = loss_3_a
    # losses["loss_d"] = loss_3_b
    # losses["loss_4"] = loss_4

    return losses


def chamfer_distance_loss_v0(model_output, mask):
    model_output_transformed = (model_output - model_output.min()) / (model_output.max() - model_output.min())
    mask_transformed = (mask - mask.min()) / (mask.max() - mask.min())
    model_output_edge_points = model_output_transformed
    U1, V1 = torch.meshgrid(torch.arange(model_output.shape[0], device=model_output.device), torch.arange(model_output.shape[1], device=model_output.device))
    U1 = U1
    V1 = V1
    U1 = U1 / (model_output.shape[1] - 1)
    V1 = V1 / (model_output.shape[0] - 1)
    T1 = model_output_edge_points
    P1 = torch.stack([U1.flatten(), V1.flatten(), T1.flatten()], dim=-1)

    U1_GT, V1_GT = torch.meshgrid(torch.arange(model_output.shape[0], device=model_output.device), torch.arange(model_output.shape[1], device=model_output.device))
    U1_GT = U1_GT
    V1_GT = V1_GT
    U1_GT = U1_GT / (model_output.shape[1] - 1)
    V1_GT = V1_GT / (model_output.shape[0] - 1)
    T1_GT = mask_transformed
    P1_GT = torch.stack([U1_GT.flatten(), V1_GT.flatten(), T1_GT.flatten()], dim=-1)

    # Make P1 and P1_GT of shape (1, X, 3)
    P1 = P1.unsqueeze(0)
    P1_GT = P1_GT.unsqueeze(0)

    # Compute the chamfer distance
    chamferDist = ChamferDistance()
    chamfer_distance_1_a = chamferDist(P1, P1_GT)
    chamfer_distance_1_b = chamferDist(P1_GT, P1)
    chamfer_distance_1 = chamfer_distance_1_a + chamfer_distance_1_b

    return chamfer_distance_1


def chamfer_distance_loss(model_output, mask):
    model_output_transformed = (model_output - model_output.min()) / (model_output.max() - model_output.min())
    edge_indices = model_output_transformed > 0.5
    if (edge_indices.sum() == 0):
        print("No edge points found")
    model_output_edge_points = model_output_transformed[edge_indices]
    U1, V1 = torch.meshgrid(torch.arange(model_output.shape[0], device=model_output.device), torch.arange(model_output.shape[1], device=model_output.device))
    U1 = U1[edge_indices]
    V1 = V1[edge_indices]
    U1 = U1 / (model_output.shape[1] - 1)
    V1 = V1 / (model_output.shape[0] - 1)
    T1 = model_output_edge_points * torch.ones_like(U1)
    P1 = torch.stack([U1, V1, T1], dim=-1)

    U1_GT, V1_GT = torch.meshgrid(torch.arange(model_output.shape[0], device=model_output.device), torch.arange(model_output.shape[1], device=model_output.device))
    U1_GT = U1_GT[mask > 0.5]
    V1_GT = V1_GT[mask > 0.5]
    U1_GT = U1_GT / (model_output.shape[1] - 1)
    V1_GT = V1_GT / (model_output.shape[0] - 1)
    T1_GT = torch.ones_like(U1_GT)
    P1_GT = torch.stack([U1_GT, V1_GT, T1_GT], dim=-1)

    # Make P1 and P1_GT of shape (1, X, 3)
    P1 = P1.unsqueeze(0)
    P1_GT = P1_GT.unsqueeze(0)

    # Compute the chamfer distance
    chamferDist = ChamferDistance()
    chamfer_distance_1_a = chamferDist(P1, P1_GT)
    chamfer_distance_1_b = chamferDist(P1_GT, P1)
    chamfer_distance_1 = chamfer_distance_1_a + chamfer_distance_1_b


    # # Do the same for the background points
    background_indices = model_output_transformed <= 0.5
    U1b, V1b = torch.meshgrid(torch.arange(model_output.shape[0], device=model_output.device), torch.arange(model_output.shape[1], device=model_output.device))
    U1b = U1b[background_indices]
    V1b = V1b[background_indices]
    U1b = U1b / (model_output.shape[1] - 1)
    V1b = V1b / (model_output.shape[0] - 1)
    T1b = model_output_transformed[background_indices]
    P1b = torch.stack([U1b, V1b, T1b], dim=-1)

    U1b_GT, V1b_GT = torch.meshgrid(torch.arange(model_output.shape[0], device=model_output.device), torch.arange(model_output.shape[1], device=model_output.device))
    U1b_GT = U1b_GT[mask <= 0.5]
    V1b_GT = V1b_GT[mask <= 0.5]
    U1b_GT = U1b_GT / (model_output.shape[1] - 1)
    V1b_GT = V1b_GT / (model_output.shape[0] - 1)
    T1b_GT = torch.zeros_like(U1b_GT)
    P1b_GT = torch.stack([U1b_GT, V1b_GT, T1b_GT], dim=-1)

    # Make P1 and P1_GT of shape (1, X, 3)
    P1b = P1b.unsqueeze(0)
    P1b_GT = P1b_GT.unsqueeze(0)

    # Compute the chamfer distance
    chamferDist = ChamferDistance()
    chamfer_distance_2_a = chamferDist(P1b, P1b_GT)
    chamfer_distance_2_b = chamferDist(P1b_GT, P1b)
    chamfer_distance_2 = chamfer_distance_2_a + chamfer_distance_2_b


    return chamfer_distance_1, chamfer_distance_2




def soft_indexing_loss_v0(model_output, mask_torch):
    output_norm = (model_output + 1) / 2
    # output_norm = (model_output - model_output.min()) / (model_output.max() - model_output.min())
    UV1 = get_uv_soft_thresholding(output_norm, epsilon=0.5)
    init_value = -1

    T1_value = 1.0
    h, w = model_output.shape[0], model_output.shape[1]

    soft_out = soft_indexing_2d(UV1, h, w, T1_value, init_value)
    soft_indexing_2d_loss = ((soft_out.flatten() - mask_torch.flatten())**2).mean()

    return soft_indexing_2d_loss
    







class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate
    
    
class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords        

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)
                
                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()
                    
                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else: 
                x = layer(x)
                
                if retain_grad:
                    x.retain_grad()
                    
            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations
    

class ImageFitting(Dataset):
    def __init__(self, sidelength):
        super().__init__()
        img = get_cameraman_tensor(sidelength)
        self.pixels = img.permute(1, 2, 0).view(-1, 1)
        self.coords = get_mgrid(sidelength, 2)

    def __len__(self):
        return 1

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
            
        return self.coords, self.pixels

    
class PoissonEqn(Dataset):
    def __init__(self, sidelength):
        super().__init__()
        img = get_cameraman_tensor(sidelength)
        
        # Compute gradient and laplacian       
        grads_x = scipy.ndimage.sobel(img.numpy(), axis=1).squeeze(0)[..., None]
        grads_y = scipy.ndimage.sobel(img.numpy(), axis=2).squeeze(0)[..., None]
        grads_x, grads_y = torch.from_numpy(grads_x), torch.from_numpy(grads_y)
                
        self.grads = torch.stack((grads_x, grads_y), dim=-1).view(-1, 2)
        self.laplace = scipy.ndimage.laplace(img.numpy()).squeeze(0)[..., None]
        self.laplace = torch.from_numpy(self.laplace)
        
        self.pixels = img.permute(1, 2, 0).view(-1, 1)
        self.coords = get_mgrid(sidelength, 2)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.coords, {'pixels':self.pixels, 'grads':self.grads, 'laplace':self.laplace}


class EDGEData(Dataset):
    def __init__(self,
                 depth_img,
                 event_bins_G_np,
                 edge_pipeline : EdgePipeline,
                 v_shift):
        super().__init__()

        device = v_shift.device

        depth_img_torch = torch.from_numpy(depth_img).to(device)
        self.event_bins_GT = torch.from_numpy(event_bins_G_np)
        self.event_bins_simulated = torch.zeros_like(self.event_bins_GT)

        event_output = edge_pipeline(depth_img_torch, v_shift)
        self.event_bins_simulated[:,:,0] = event_output["event_bin_0"]

        event_bins_pred = event_output["event_bins"]
        for i in range(self.event_bins_GT.shape[-1] - 1):
            U1 = event_bins_pred[i]["U1"]
            V1 = event_bins_pred[i]["V1"]
            Z1 = event_bins_pred[i]["Z1"] 
            T1 = event_bins_pred[i]["T1"] * torch.ones_like(U1)
            P1 = torch.stack([U1, V1, T1], dim=-1)

            event_img = get_event_image(U1, V1, T1, depth_img_torch)
            self.event_bins_simulated[:,:,i+1] = event_img


        self.coords = get_mgrid(depth_img.squeeze().shape, 2)
        
        self.depth_GT = depth_img_torch.view(-1, 1)
        self.depth_event_GT_view = get_event_image(U1, V1, Z1, depth_img_torch)
        

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.coords, {'depth_GT':self.depth_GT, 'event_bins_GT':self.event_bins_GT, 'event_bins_simulated':self.event_bins_simulated}