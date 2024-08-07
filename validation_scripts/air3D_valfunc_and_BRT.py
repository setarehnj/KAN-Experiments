import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, utils, training, loss_functions, modules, diff_operators

import torch
import numpy as np
import math
from torch.utils.data import DataLoader
import configargparse
import scipy.io as spio
from sklearn.metrics import mean_squared_error

logging_root = './logs/Wednesday24'
os.makedirs(logging_root, exist_ok=True)
angle_alpha = 1.2

# Setting to plot
ckpt_path = './Deepreach_trained_checkpoints/air3D_ckpt.pth'
model = modules.SingleBVPNet(in_features=4, out_features=1, type=activation, mode='mlp',
                             final_layer_factor=1., hidden_features=512, num_hidden_layers=3)
model.cuda()
checkpoint = torch.load(ckpt_path)
try:
  model_weights = checkpoint['model']
except:
  model_weights = checkpoint
model.load_state_dict(model_weights)
model.eval()

# Load the ground truth BRS data
true_BRT_path = './Deepreach_trained_checkpoints/analytical_BRT_air3D.mat'
true_data = spio.loadmat(true_BRT_path)

activation = 'sine'

# Use all time steps and theta values from the data
times = np.linspace(0, 1, 11)  # 11 time steps from 0 to 1
time_indices_matlab = [int(time/0.1) + 1 for time in times]
thetas = true_data['gmat'][0, 0, :, 2]  # All 101 theta values

num_times = len(times)  # This will be 11
num_thetas = len(thetas)  # This will be 101

def val_fn_BRS(model):
    # Initialize arrays to store results
    valfunc_all = np.zeros((num_times, num_thetas, 101, 101))
    valfunc_true_all = np.zeros((num_times, num_thetas, 101, 101))
    brs_predicted_all = np.zeros((num_times, num_thetas, 101, 101))
    brs_actual_all = np.zeros((num_times, num_thetas, 101, 101))
    mse_brs_list = []
    mse_valfunc_list = []

    for i in range(num_times):
        for j in range(num_thetas):
            state_coords = torch.tensor(np.reshape(true_data['gmat'][:, :, j, :], (-1, 3)), dtype=torch.float32)
            state_coords[:, 2] = state_coords[:, 2] / (angle_alpha * math.pi)
            time_coords = torch.ones(state_coords.shape[0], 1) * times[i]
            coords = torch.cat((time_coords, state_coords), dim=1)[None]
            
            # Compute the value function
            model_in = {'coords': coords.cuda()}
            model_out = model(model_in)

            # Detach outputs and reshape
            valfunc = model_out['model_out'].detach().cpu().numpy()
            valfunc_true = true_data['data'][:, :, j, time_indices_matlab[i]]
            valfunc = np.reshape(valfunc, valfunc_true.shape)

            # Unnormalize the value function
            norm_to = 0.02
            mean = 0.25
            var = 0.5
            valfunc = (valfunc*var/norm_to) + mean 

            # Compute BRS
            brs_predicted = (valfunc <= 0.001) * 1.
            brs_actual = (valfunc_true <= 0.001) * 1.

            # Store results
            valfunc_all[i, j] = valfunc
            valfunc_true_all[i, j] = valfunc_true
            brs_predicted_all[i, j] = brs_predicted
            brs_actual_all[i, j] = brs_actual

            mse_brs = mean_squared_error(brs_actual, brs_predicted)
            mse_valfunc = mean_squared_error(valfunc_true, valfunc)
            mse_brs_list.append(mse_brs)
            mse_valfunc_list.append(mse_valfunc)
            print(f"Time: {times[i]}, Theta: {thetas[j]}, MSE BRS: {mse_brs:.9f}, MSE Valfunc: {mse_valfunc:.9f}")
            

    avg_mse_brs = np.mean(mse_brs_list)
    avg_mse_valfunc = np.mean(mse_valfunc_list)
    print(f"Average MSE for BRT (w.r.t. analytical solution): {avg_mse_brs:.9f}")
    print(f"Average MSE for Value Function (w.r.t. LST solution): {avg_mse_valfunc:.9f}")

    return valfunc_all, valfunc_true_all, brs_predicted_all, brs_actual_all, avg_mse_brs, avg_mse_valfunc

# Run the validation
valfunc_all, valfunc_true_all, brs_predicted_all, brs_actual_all, avg_mse_brs, avg_mse_valfunc = val_fn_BRS(model)

print(f"Average MSE for BRT (w.r.t. analytical solution): {avg_mse_brs:.9f}")
print(f"Average MSE for Value Function (w.r.t. LST solution): {avg_mse_valfunc:.9f}")

# Plot BRT for all time steps
fig_brt, axes = plt.subplots(3, 4, figsize=(20, 15))
axes = axes.flatten()

for i in range(num_times):
    ax = axes[i]
    brs_predicted = np.max(brs_predicted_all[i], axis=0)
    brs_actual = np.max(brs_actual_all[i], axis=0)
    
    ax.imshow(brs_predicted.T, cmap='bwr', origin='lower', vmin=-1., vmax=1., extent=(-1., 1., -1., 1.), interpolation='bilinear')
    ax.imshow(brs_actual.T, cmap='seismic', alpha=0.5, origin='lower', vmin=-1., vmax=1., extent=(-1., 1., -1., 1.), interpolation='bilinear')
    ax.set_title(f't = {times[i]:.2f}')

fig_brt.suptitle('BRT Comparison for All Time Steps')
fig_brt.tight_layout()
fig_brt.savefig(os.path.join(logging_root, 'Air3D_BRT_comparison_all_times.png'))

# Plot value functions for all time steps
def plot_value_functions(valfunc_true_all, valfunc_all, times):
    fig_valfunc_LS, axes_LS = plt.subplots(3, 4, figsize=(20, 15))
    fig_valfunc_siren, axes_siren = plt.subplots(3, 4, figsize=(20, 15))
    axes_LS = axes_LS.flatten()
    axes_siren = axes_siren.flatten()

    for i in range(len(times)):
        valfunc_true = np.max(valfunc_true_all[i], axis=0)
        valfunc = np.max(valfunc_all[i], axis=0)
        
        ax_LS = axes_LS[i]
        im_LS = ax_LS.imshow(valfunc_true.T, cmap='bwr', origin='lower', extent=(-1., 1., -1., 1.), vmin=-0.25, vmax=1.2)
        ax_LS.set_title(f't = {times[i]:.2f}')
        fig_valfunc_LS.colorbar(im_LS, ax=ax_LS)
        
        ax_siren = axes_siren[i]
        im_siren = ax_siren.imshow(valfunc.T, cmap='bwr', origin='lower', extent=(-1., 1., -1., 1.), vmin=-0.25, vmax=1.2)
        ax_siren.set_title(f't = {times[i]:.2f}')
        fig_valfunc_siren.colorbar(im_siren, ax=ax_siren)

    fig_valfunc_LS.suptitle('LS Value Function for All Time Steps')
    fig_valfunc_siren.suptitle('SIREN Value Function for All Time Steps')
    fig_valfunc_LS.tight_layout()
    fig_valfunc_siren.tight_layout()
    
    return fig_valfunc_LS, fig_valfunc_siren

fig_valfunc_LS, fig_valfunc_siren = plot_value_functions(valfunc_true_all, valfunc_all, times)
fig_valfunc_LS.savefig(os.path.join(logging_root, 'Air3D_LS_valfunc_all_times.png'))
fig_valfunc_siren.savefig(os.path.join(logging_root, 'Air3D_Siren_valfunc_all_times.png'))

# Save data
val_functions = {
    'LS': valfunc_true_all,
    'siren': valfunc_all,
    'BRS_predicted': brs_predicted_all,
    'BRS_actual': brs_actual_all,
    'times': times,
    'thetas': thetas
}
spio.savemat(os.path.join(logging_root, 'Air3D_raw_valfuncs_and_brs_all_times.mat'), val_functions)