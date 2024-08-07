
import importlib
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dataio, utils, training, loss_functions, modules
importlib.reload(dataio)
importlib.reload(utils)
importlib.reload(training)
importlib.reload(loss_functions)
importlib.reload(modules)
import torch
import random
import numpy as np
import math
from torch.utils.data import DataLoader
import configargparse
from training import ModelLightningModule
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
import pandas as pd
import time
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pytorch_lightning.callbacks import ModelCheckpoint



###===========================================================================================================================

# def ensure_dir(path):
#     os.makedirs(path, exist_ok= True)


# def plot_metric(lightning_modules, metric_name, root_path, log_scale=False):
#     plot_dir = os.path.join(root_path, 'plots', metric_name.replace('_', ' ').title())
#     ensure_dir(plot_dir)
    
#     plt.figure(figsize=(10, 6))
#     for lightning_module in lightning_modules:
#       print(f'Metric name is: {metric_name}')
#       # print(f'Metric shape is: {getattr(lightning_module, metric_name).shape}')
#       data = getattr(lightning_module, metric_name)
#       epochs = range(0, len(data), 1)
#       values = data[::1]
#       plt.plot(epochs, values, label=lightning_module.model_name)
    
#     if log_scale:
#         plt.yscale('log')
#     plt.xlabel('Epochs')
#     plt.ylabel(metric_name.replace('_', ' ').title())
#     plt.legend()
#     plt.title(f'{metric_name.replace("_", " ").title()} vs Epochs')
#     filename = f'{metric_name}_plot.png'
#     plt.savefig(os.path.join(plot_dir, filename))
#     plt.close()

# def plot_metric_intervals(lightning_modules, metric_name, root_path, log_scale=False, final_end_epoch =  max([len(getattr(lm, metric_name))] for lm in lightning_module)s, interval=100):
#     plot_dir = os.path.join(root_path, 'plots', metric_name.replace('_', ' ').title(), 'intervals')
#     ensure_dir(plot_dir)
    
#     for start_epoch in range(0, final_end_epoch, interval):
#         plt.figure(figsize=(10, 6))
#         for lightning_module in lightning_modules:
#             data = getattr(lightning_module, metric_name)
#             end_epoch = min(start_epoch + interval, len(data))
#             epochs = range(start_epoch, end_epoch)
#             values = data[start_epoch:end_epoch]
#             plt.plot(epochs, values, label=lightning_module.model_name)
        
#         if log_scale:
#             plt.yscale('log')
#         plt.xlabel('Epochs')
#         plt.ylabel(metric_name.replace('_', ' ').title())
#         plt.legend()
#         plt.title(f'{metric_name.replace("_", " ").title()} vs Epochs [{start_epoch}, {end_epoch - 1}]')
#         filename = f'{metric_name}_plot_epochs_{start_epoch}_{end_epoch - 1}.png'
#         plt.savefig(os.path.join(plot_dir, filename))
#         plt.close()

# def plot_all_metrics(lightning_modules, root_path):
#     metrics = [
#         ('avg_losses', True),
#         ('iteration_times', False),
#         ('total_training_times', False), 
#         ('train_losses', True)
#     ]
#     for metric, log_scale in metrics:
#         plot_metric(lightning_modules, metric, root_path, log_scale)
#         plot_metric_intervals(lightning_modules, metric, root_path, log_scale, interval=1000)
       
#     if metric == 'iteration_times':
#         plot_metric_intervals(lightning_modules, metric, root_path, log_scale, final_end_epoch = 100, interval=100)
#         plot_metric_intervals(lightning_modules, metric, root_path, log_scale, final_end_epoch = 20, interval=5)



def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def plot_metric(lightning_modules, metric_name, root_path, log_scale=False):
    plot_dir = os.path.join(root_path, 'plots', metric_name.replace('_', ' ').title())
    ensure_dir(plot_dir)
    
    plt.figure(figsize=(10, 6))
    for lightning_module in lightning_modules:
        print(f'Metric name is: {metric_name}')
        # print(f'Metric shape is: {getattr(lightning_module, metric_name).shape}')
        data = getattr(lightning_module, metric_name)
        epochs = range(0, len(data), 1)
        values = data[::1]
        plt.plot(epochs, values, label=lightning_module.model_name)
    
    if log_scale:
        plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name.replace('_', ' ').title())
    plt.legend()
    plt.title(f'{metric_name.replace("_", " ").title()} vs Epochs')
    filename = f'{metric_name}_plot.png'
    plt.savefig(os.path.join(plot_dir, filename))
    plt.close()

def plot_metric_intervals(lightning_modules, metric_name, root_path, log_scale=False, final_end_epoch=None, interval=100):
    if final_end_epoch is None:
        final_end_epoch = max([len(getattr(lm, metric_name)) for lm in lightning_modules])
    
    plot_dir = os.path.join(root_path, 'plots', metric_name.replace('_', ' ').title(), 'intervals')
    ensure_dir(plot_dir)
    
    for start_epoch in range(0, final_end_epoch, interval):
        plt.figure(figsize=(10, 6))
        for lightning_module in lightning_modules:
            data = getattr(lightning_module, metric_name)
            end_epoch = min(start_epoch + interval, len(data))
            epochs = range(start_epoch, end_epoch)
            values = data[start_epoch:end_epoch]
            plt.plot(epochs, values, label=lightning_module.model_name)
        
        if log_scale:
            plt.yscale('log')
        plt.xlabel('Epochs')
        plt.ylabel(metric_name.replace('_', ' ').title())
        plt.legend()
        plt.title(f'{metric_name.replace("_", " ").title()} vs Epochs [{start_epoch}, {end_epoch - 1}]')
        filename = f'{metric_name}_plot_epochs_{start_epoch}_{end_epoch - 1}.png'
        plt.savefig(os.path.join(plot_dir, filename))
        plt.close()

def plot_all_metrics(lightning_modules, root_path):
    metrics = [
        ('avg_losses', True),
        ('iteration_times', False),
        ('total_training_times', False), 
        ('train_losses', True)
    ]
    for metric, log_scale in metrics:
        plot_metric(lightning_modules, metric, root_path, log_scale)
        plot_metric_intervals(lightning_modules, metric, root_path, log_scale, interval=1000)
       
        if metric == 'iteration_times':
            plot_metric_intervals(lightning_modules, metric, root_path, log_scale, final_end_epoch=1000, interval=100)
            plot_metric_intervals(lightning_modules, metric, root_path, log_scale, final_end_epoch=100, interval=20)
            plot_metric_intervals(lightning_modules, metric, root_path, log_scale, final_end_epoch=20, interval=5)

    
    

# Usage (after training is complete):
def gpu_warmup(device, input_size, hidden_size, output_size, num_iterations=10):
    # Create a dummy input tensor
    dummy_input = torch.randn(64, input_size, device=device)

    # Create a simple model with linear layers
    model = torch.nn.Sequential(
        torch.nn.Linear(input_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, output_size)
    ).to(device)

    # Perform multiple forward passes to warm up the GPU
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model(dummy_input)

    # Clear the GPU cache
    torch.cuda.empty_cache()

    print("GPU warm-up completed")


###===========================================================================================================================
print("PyTorch CUDA available:", torch.cuda.is_available())
print("PyTorch CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("PyTorch CUDA device name:", torch.cuda.get_device_name(0))


seed_everything(42, workers=True)

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
p.add_argument('--fourier_num_hl', type=int, default=4, required=False, help='The number of hidden layers')
p.add_argument('--fourier_num_nl', type=int, default=40, required=False, help='Number of neurons per hidden layer.')
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



dataloader = DataLoader(dataset, shuffle=False, batch_size=opt.batch_size, pin_memory=True, num_workers=0)

# Define models for Chebyshev, MLP, Fourier, and Wavelet
#chebyshev_model = modules.SingleBVPNetwithKAN(in_features=4, out_features=1, final_layer_factor=1., hidden_features=opt.chebyshev_num_nl, num_hidden_layers=opt.chebyshev_num_hl)
dnn_model = modules.SingleBVPNetwithDNN(in_features=4, out_features=1, final_layer_factor=1., hidden_features=opt.dnn_num_nl, num_hidden_layers=opt.dnn_num_hl)
#fourier_model = modules.SingleBVPNetwithFourier(in_features=4, out_features=1, final_layer_factor=1., hidden_features=opt.fourier_num_nl, num_hidden_layers=opt.fourier_num_hl)
#wavelet_model = modules.SingleBVPNetwithWavelet(in_features=4, out_features=1, final_layer_factor=1., hidden_features=opt.wavelet_num_nl, num_hidden_layers=opt.wavelet_num_hl)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f"Total parameters in MLP model: {count_parameters(dnn_model)}")


loss_fn = loss_functions.initialize_hji_air3D(dataset, opt.minWith)


root_path = os.path.join(opt.logging_root, opt.experiment_name)

tensorboard_dir = "Aug6_all_models"
# Create the main logging directory
os.makedirs(root_path, exist_ok=True)
    
# Create the TensorBoard logging directory
tensorboard_path = os.path.join(root_path, tensorboard_dir)
os.makedirs(tensorboard_path, exist_ok=True)
dnn_logger = TensorBoardLogger(save_dir=tensorboard_path, name="dnn")


# Create the lightning modules
dnn_lightning = ModelLightningModule(dnn_model, loss_fn, opt.lr, model_name='dnn', baseline_loss=322.0)

checkpoint_callback = ModelCheckpoint(
    dirpath=os.path.join(root_path, 'checkpoints'),
    filename='MLP-{epoch:02d}-{train_loss_epoch:.2f}',
    save_top_k=10,
    verbose=True,
    save_last=True,
    monitor='train_loss_epoch',
    mode='min',
    every_n_epochs = 20000
)

torch.cuda.empty_cache()
# Define the trainers
trainer_dnn = pl.Trainer(max_epochs=opt.dnn_num_epochs, accelerator='gpu',devices = [1], logger= dnn_logger, log_every_n_steps=1, callbacks=[checkpoint_callback])

# Train the models
trainer_dnn.fit(dnn_lightning, dataloader)

