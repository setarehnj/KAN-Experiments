import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dataio, utils, training, loss_functions, modules
import torch
import random
import numpy as np
import math
from torch.utils.data import DataLoader
import configargparse
from training import ModelLightningModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import pandas as pd
import time
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

###===========================================================================================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok= True)

def plot_metric(lightning_modules, metric_name, root_path, log_scale=False, interval=1):
    plot_dir = os.path.join(root_path, 'plots', metric_name.replace('_', ' ').title())
    ensure_dir(plot_dir)
    
    plt.figure(figsize=(10, 6))
    for lightning_module in lightning_modules:
        data = getattr(lightning_module, metric_name)
        epochs = range(0, len(data), interval)
        values = data[::interval]
        plt.plot(epochs, values, label=lightning_module.model_name)
    
    if log_scale:
        plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name.replace('_', ' ').title())
    plt.legend()
    plt.title(f'{metric_name.replace("_", " ").title()} vs Epochs')
    if interval == 1:
        filename = f'{metric_name}_plot.png'
    else:
        filename = f'{metric_name}_plot_interval_{interval}.png'
    plt.savefig(os.path.join(plot_dir, filename))
    plt.close()

def plot_all_metrics(lightning_modules, root_path):
    metrics = [
        ('train_losses', True),
        ('iteration_times', False),
        ('total_training_times', False),
        ('total_training_losses', True)
    ]
    
    for metric, log_scale in metrics:
        plot_metric(lightning_modules, metric, root_path, log_scale)
        plot_metric(lightning_modules, metric, root_path, log_scale, interval=100)
        if metric == 'train_losses':
            plot_metric(lightning_modules, metric, root_path, log_scale, interval=1000)
    
    

# Usage (after training is complete):



###===========================================================================================================================
print("PyTorch CUDA available:", torch.cuda.is_available())
print("PyTorch CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("PyTorch CUDA device name:", torch.cuda.get_device_name(0))
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=32)
p.add_argument('--lr', type=float, default=2e-5, help='learning rate. default=2e-5')
p.add_argument('--dnn_num_epochs', type=int, default=120000, help='Number of epochs to train for.')
p.add_argument('--chebyshev_num_epochs', type=int, default=120000, help='Number of epochs to train for.')
p.add_argument('--fourier_num_epochs', type=int, default=120000, help='Number of epochs to train for.')
p.add_argument('--wavelet_num_epochs', type=int, default=120000, help='Number of epochs to train for.')
p.add_argument('--epochs_til_ckpt', type=int, default=1000, help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=100, help='Time interval in seconds until tensorboard summary is saved.')
p.add_argument('--model', type=str, default='sine', required=False, choices=['sine', 'tanh', 'sigmoid', 'relu'],
               help='Type of model to evaluate, default is sine.')
p.add_argument('--tMin', type=float, default=0.0, required=False, help='Start time of the simulation')
p.add_argument('--tMax', type=float, default=0.5, required=False, help='End time of the simulation')
p.add_argument('--dnn_num_hl', type=int, default=3, required=False, help='The number of hidden layers')
p.add_argument('--dnn_num_nl', type=int, default=512, required=False, help='Number of neurons per hidden layer.')
p.add_argument('--chebyshev_num_hl', type=int, default=3, required=False, help='The number of hidden layers')
p.add_argument('--chebyshev_num_nl', type=int, default=32, required=False, help='Number of neurons per hidden layer.')
p.add_argument('--fourier_num_hl', type=int, default=3, required=False, help='The number of hidden layers')
p.add_argument('--fourier_num_nl', type=int, default=32, required=False, help='Number of neurons per hidden layer.')
p.add_argument('--wavelet_num_hl', type=int, default=3, required=False, help='The number of hidden layers')
p.add_argument('--wavelet_num_nl', type=int, default=32, required=False, help='Number of neurons per hidden layer.')
p.add_argument('--pretrain_iters', type=int, default=2000, required=False, help='Number of pretrain iterations')
p.add_argument('--counter_start', type=int, default=-1, required=False, help='Defines the initial time for the curriculul training')
p.add_argument('--counter_end', type=int, default=-1, required=False, help='Defines the linear step for curriculum training starting from the initial time')
p.add_argument('--num_src_samples', type=int, default=1000, required=False, help='Number of source samples at each time step')

p.add_argument('--velocity', type=float, default=0.6, required=False, help='Speed of the dubins car')
p.add_argument('--omega_max', type=float, default=1.1, required=False, help='Turn rate of the car')
p.add_argument('--angle_alpha', type=float, default=1.0, required=False, help='Angle alpha coefficient.')
p.add_argument('--collisionR', type=float, default=0.25, required=False, help='Collision radius between vehicles')
p.add_argument('--minWith', type=str, default='none', required=False, choices=['none', 'zero', 'target'], help='BRS vs BRT computation')

p.add_argument('--clip_grad', default=0.0, type=float, help='Clip gradient.')
p.add_argument('--use_lbfgs', default=False, type=bool, help='use L-BFGS.')
p.add_argument('--pretrain', action='store_true', default=False, required=False, help='Pretrain dirichlet conditions')

p.add_argument('--seed', type=int, default=0, required=False, help='Seed for the simulation.')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
p.add_argument('--checkpoint_toload', type=int, default=0, help='Checkpoint from which to restart the training.')
opt = p.parse_args()
torch.cuda.empty_cache()

# Set the source coordinates for the target set and the obstacle sets
source_coords = [0., 0., 0.]
if opt.counter_start == -1:
    opt.counter_start = opt.checkpoint_toload

if opt.counter_end == -1:
    opt.counter_end = opt.dnn_num_epochs

dataset = dataio.ReachabilityAir3DSource(numpoints=65000, collisionR=opt.collisionR, velocity=opt.velocity,
                                         omega_max=opt.omega_max, pretrain=opt.pretrain, tMin=opt.tMin,
                                         tMax=opt.tMax, counter_start=opt.counter_start, counter_end=opt.counter_end,
                                         pretrain_iters=opt.pretrain_iters, seed=opt.seed,
                                         angle_alpha=opt.angle_alpha,
                                         num_src_samples=opt.num_src_samples)

root_path = os.path.join(opt.logging_root, opt.experiment_name)

dataloader = DataLoader(dataset, shuffle=False, batch_size=opt.batch_size, pin_memory=True, num_workers=0)

# Define models for Chebyshev, MLP, Fourier, and Wavelet
chebyshev_model = modules.SingleBVPNetwithKAN(in_features=4, out_features=1, final_layer_factor=1., hidden_features=opt.chebyshev_num_nl, num_hidden_layers=opt.chebyshev_num_hl)
dnn_model = modules.SingleBVPNetwithDNN(in_features=4, out_features=1, final_layer_factor=1., hidden_features=opt.dnn_num_nl, num_hidden_layers=opt.dnn_num_hl)
fourier_model = modules.SingleBVPNetwithFourier(in_features=4, out_features=1, final_layer_factor=1., hidden_features=opt.fourier_num_nl, num_hidden_layers=opt.fourier_num_hl)
wavelet_model = modules.SingleBVPNetwithWavelet(in_features=4, out_features=1, final_layer_factor=1., hidden_features=opt.wavelet_num_nl, num_hidden_layers=opt.wavelet_num_hl)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters in Chebyshev model: {count_parameters(chebyshev_model)}")

print(f"Total parameters in MLP model: {count_parameters(dnn_model)}")
print(f"Total parameters in Fourier model: {count_parameters(fourier_model)}")
print(f"Total parameters in Wavelet model: {count_parameters(wavelet_model)}")


loss_fn = loss_functions.initialize_hji_air3D(dataset, opt.minWith)

root_path = os.path.join(opt.logging_root, opt.experiment_name)


chebyshev_logger = TensorBoardLogger("today_logs", name="chebyshev")
dnn_logger = TensorBoardLogger("today_logs", name="dnn")
fourier_logger = TensorBoardLogger("today_logs", name="fourier")
wavelet_logger = TensorBoardLogger("today_logs", name="wavelet")


chebyshev_lightning = ModelLightningModule(chebyshev_model, loss_fn, opt.lr, opt.steps_til_summary, model_name='chebyshev')
dnn_lightning = ModelLightningModule(dnn_model, loss_fn, opt.lr, opt.steps_til_summary, model_name='dnn')
fourier_lightning = ModelLightningModule(fourier_model, loss_fn, opt.lr, opt.steps_til_summary, model_name='fourier')
wavelet_lightning = ModelLightningModule(wavelet_model, loss_fn, opt.lr, opt.steps_til_summary, model_name='wavelet')


# Define the trainers
trainer_chebyshev = pl.Trainer(max_epochs=opt.chebyshev_num_epochs, accelerator='gpu', devices=[0], logger=chebyshev_logger, log_every_n_steps=1)
trainer_dnn = pl.Trainer(max_epochs=opt.dnn_num_epochs, accelerator='gpu', devices=[0], logger= dnn_logger, log_every_n_steps=1)
trainer_fourier = pl.Trainer(max_epochs=opt.fourier_num_epochs, accelerator='gpu', devices=[1], logger=fourier_logger, log_every_n_steps=1)
trainer_wavelet = pl.Trainer(max_epochs=opt.wavelet_num_epochs, accelerator='gpu', devices=[1], logger=wavelet_logger, log_every_n_steps=1)


# # Define the trainers
# trainer_chebyshev = pl.Trainer(max_epochs=opt.chebyshev_num_epochs, accelerator='gpu', devices=[0], default_root_dir='fri_logs', logger=logger, log_every_n_steps=1)
# trainer_dnn = pl.Trainer(max_epochs=opt.dnn_num_epochs, accelerator='gpu', devices=[0], default_root_dir='fri_logs', logger=logger, log_every_n_steps=1)
# trainer_fourier = pl.Trainer(max_epochs=opt.fourier_num_epochs, accelerator='gpu', devices=[1], default_root_dir='fri_logs', logger=logger, log_every_n_steps=1)
# trainer_wavelet = pl.Trainer(max_epochs=opt.wavelet_num_epochs, accelerator='gpu', devices=[1], default_root_dir='fri_logs', logger=logger, log_every_n_steps=1)

# Train the models
trainer_chebyshev.fit(chebyshev_lightning, dataloader)
trainer_dnn.fit(dnn_lightning, dataloader)
trainer_fourier.fit(fourier_lightning, dataloader)
trainer_wavelet.fit(wavelet_lightning, dataloader)

# Plot convergence curves
# plot_convergence_curves(chebyshev_lightning.losses, dnn_lightning.losses, fourier_lightning.losses, wavelet_lightning.losses, root_path)

# # Plot iteration time vs. epochs
# plt.figure(figsize=(10, 6))
# for lightning_module, label in zip([chebyshev_lightning, dnn_lightning, fourier_lightning, wavelet_lightning], ['chebyshev', 'dnn', 'fourier', 'wavelet']):
#     epochs = range(len(lightning_module.iteration_times))
#     plt.plot(epochs, lightning_module.iteration_times, label=label)

# plt.xlabel('Epochs')
# plt.ylabel('Iteration Time (s)')
# plt.legend()
# plt.title('Iteration Time vs Epochs')
# plt.show()

# Save scalar data to CSV
# data = {
#     'chebyshev_iteration_time': chebyshev_lightning.iteration_times,
#     'chebyshev_total_training_time': chebyshev_lightning.total_training_times,
#     'chebyshev_losses': chebyshev_lightning.losses,
#     'dnn_iteration_time': dnn_lightning.iteration_times,
#     'dnn_total_training_time': dnn_lightning.total_training_times,
#     'dnn_losses': dnn_lightning.losses,
#     'fourier_iteration_time': fourier_lightning.iteration_times,
#     'fourier_total_training_time': fourier_lightning.total_training_times,
#     'fourier_losses': fourier_lightning.losses,
#     'wavelet_iteration_time': wavelet_lightning.iteration_times,
#     'wavelet_total_training_time': wavelet_lightning.total_training_times,
#     'wavelet_losses': wavelet_lightning.losses
# }

# df = pd.DataFrame(data)
# df.to_csv(os.path.join(root_path, 'training_metrics.csv'), index=False)



# metric_names = ['train_loss_epoch', 'iteration_time', 'total_training_time', 'total_training_loss']

# # Collect and plot data
# data = collect_data_from_logger(log_dir, metric_names)

# plot_metric(data['train_loss_epoch'], root_path, 'Training Loss', log_scale=True)
# plot_metric(data['iteration_time'], root_path, 'Iteration Time')
# plot_metric(data['total_training_time'], root_path, 'Total Training Time')
# plot_metric(data['total_training_loss'], root_path, 'Total Training Loss', log_scale=True)

# # Optional: Plot specific epochs (e.g., every 100th epoch)
# def plot_specific_epochs(metric_dict, root_path, metric_name, epoch_interval=100, log_scale=False):
#     plot_dir = os.path.join(root_path, f'{metric_name.replace(" ", "_")}_interval')
#     ensure_dir(plot_dir)
    
#     plt.figure(figsize=(10, 6))
#     for model_name, values in metric_dict.items():
#         epochs = list(range(0, len(values), epoch_interval))
#         selected_values = values[::epoch_interval]
#         plt.plot(epochs, selected_values, label=model_name, marker='o')
    
#     if log_scale:
#         plt.yscale('log')
#     plt.xlabel('Epochs')
#     plt.ylabel(metric_name)
#     plt.title(f'{metric_name} (Every {epoch_interval} Epochs)')
#     plt.legend()
#     plt.savefig(os.path.join(plot_dir, f'{metric_name.replace(" ", "_")}_interval.png'))
#     plt.close()

# # Plot metrics at specific intervals
# plot_specific_epochs(data['total_training_time'], root_path, 'Total Training Time', epoch_interval=100)
# plot_specific_epochs(data['total_training_loss'], root_path, 'Total Training Loss', epoch_interval=100, log_scale=True)

# # Plot all metrics


lightning_modules = [chebyshev_lightning, dnn_lightning, fourier_lightning, wavelet_lightning]

plot_all_metrics(lightning_modules, root_path)

