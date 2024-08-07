# import torch
# import pytorch_lightning as pl
# from torch.utils.tensorboard import SummaryWriter
# import time
# import os
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# from pytorch_lightning.loggers import TensorBoardLogger



# class ModelLightningModule(pl.LightningModule):
#     def __init__(self, model, loss_fn, lr, steps_til_summary, model_name, validation_fn=None, **kwargs):
#         super(ModelLightningModule, self).__init__()
#         self.model = model
#         self.loss_fn = loss_fn
#         self.lr = lr
#         self.steps_til_summary = steps_til_summary
#         self.model_name = model_name
#         self.validation_fn = validation_fn
#         self.epoch_start_time = None
#         self.epoch_end_time = None
#         self.train_losses = []
#         self.total_training_times = []
#         self.iteration_times = []
#         self.avg_losses = []
#         self.batch_size = 1
      


#     def forward(self, x):
#         return self.model(x)

#     def on_train_epoch_start(self):
#         self.epoch_start_time = time.time()

#     def training_step(self, batch, batch_idx):
#         model_input, gt = batch
#         model_output = self(model_input)
#         losses = self.loss_fn(model_output, gt)
#         train_loss = 0
#         train_loss = sum(loss.mean() for loss in losses.values())      
#         self.train_losses.append(train_loss.item())
#         self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, logger=True)
#         return train_loss

#     def on_train_epoch_end(self):
#         self.epoch_end_time = time.time()
#         epoch_training_time = self.epoch_end_time - self.epoch_start_time
#         self.iteration_times.append(epoch_training_time)
#         total_time = sum(self.iteration_times)
#         self.total_training_times.append(total_time)
#         # Calculate and store average loss for the epoch
#         #print(f"avg_losses len: {len(self.train_losses)}")
#         self.batch_size = len(self.train_losses)
#         avg_loss = sum(self.train_losses) / self.batch_size
#         self.avg_losses.append(avg_loss)
#         #print(f'avg_losses len: {len(self.avg_losses)}')
#         self.log('epoch_training_time', epoch_training_time, on_epoch=True, prog_bar=True, sync_dist = True, logger=True)
#         self.log('total_training_time', total_time, on_epoch=True, prog_bar=True,logger=True)
#         self.log('avg_train_loss_epoch', avg_loss, on_epoch=True, prog_bar=True,logger=True)

#     def on_train_end(self):
#         checkpoint_dir = os.path.join(self.logger.log_dir, 'checkpoints')
#         os.makedirs(checkpoint_dir, exist_ok=True)
#         final_checkpoint_path = os.path.join(checkpoint_dir, f"{self.model_name}_final.ckpt")
#         torch.save(self.model.state_dict(), final_checkpoint_path)

#         final_epoch_training_loss_text = f'Model: {self.model_name}, Final Epoch Training Loss: {self.avg_losses[-1]}'
#         final_total_training_time_text = f'Model: {self.model_name}, Final Total Training Time: {self.total_training_times[-1]}'
#         self.logger.experiment.add_text('Final Epoch Training Loss', final_epoch_training_loss_text, self.current_epoch)
#         self.logger.experiment.add_text('Final Total Training Time', final_total_training_time_text, self.current_epoch)

#     def validation_step(self, batch, batch_idx):
#         if self.validation_fn:
#             return self.validation_fn(self, batch, batch_idx)

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
#         return optimizer

# class LinearDecaySchedule():
#     def __init__(self, start_val, final_val, num_steps):
#         self.start_val = start_val
#         self.final_val = final_val
#         self.num_steps = num_steps

#     def __call__(self, iter):
#         return self.start_val + (self.final_val - self.start_val) * min(iter / self.num_steps, 1.)


import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import os
import time


class ModelLightningModule(pl.LightningModule):
    def __init__(self, model, loss_fn, lr, model_name, baseline_loss=float('inf')):
        super(ModelLightningModule, self).__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.model_name = model_name
        self.baseline_loss = baseline_loss
        self.total_train_loss_epoch = []
        self.single_losses_epoch = {}
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.epoch_start_time = None
        self.epoch_end_time = None
        self.total_training_time = []
        self.epoch_training_time = []
        self.current_epoch_losses = None  

    def forward(self, x):
        return self.model(x)

    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()

    def training_step(self, batch, batch_idx):
        model_input, gt = batch
        model_output = self(model_input)
        losses = self.loss_fn(model_output, gt)
        total_loss = sum(loss.mean() for loss in losses.values())
        self.total_train_loss_epoch.append(total_loss.item())
        self.current_epoch_losses = {name: value.mean().item() for name, value in losses.items()}
        return total_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def save_checkpoint(self, filename):
        checkpoint_path = os.path.join(self.logger.log_dir, 'checkpoints', filename)
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(self.state_dict(), checkpoint_path)
    

    def on_train_epoch_end(self):
        self.epoch_end_time = time.time()
        epoch_training_time = self.epoch_end_time - self.epoch_start_time
        self.epoch_training_time.append(epoch_training_time)
        total_time = sum(self.epoch_training_time)
        self.total_training_time.append(total_time) 
        for name, value in self.current_epoch_losses.items():
            if name not in self.single_losses_epoch:
                self.single_losses_epoch[name] = []
            self.single_losses_epoch[name].append(value)
            self.log(f'{name}_loss', value, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss_epoch',self.total_train_loss_epoch[-1], on_epoch=True, prog_bar=True,logger=True)
        self.log('epoch_training_time', epoch_training_time, on_epoch=True, prog_bar=True, logger=True)
        self.log('total_training_time', total_time, on_epoch=True, prog_bar=True,logger=True)
        if self.current_epoch % 1000 == 0: 
            checkpoint_filename = f"{self.model_name}_epoch_{self.current_epoch}.ckpt"
            self.save_checkpoint(checkpoint_filename)
            #self.plot_value_function(self.current_epoch) 

    def plot_value_function(self, epoch):
        ckpt_dir = os.path.join(self.logger.log_dir, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)

        times = [0., 0.5*(self.opt.tMax - 0.1), (self.opt.tMax - 0.1)]
        num_times = len(times)
        thetas = [-math.pi, -0.5*math.pi, 0., 0.5*math.pi, math.pi]
        num_thetas = len(thetas)
        fig = plt.figure(figsize=(5*num_times, 5*num_thetas))
        sidelen = 200
        mgrid_coords = self.dataio.get_mgrid(sidelen)

        for i in range(num_times):
            time_coords = torch.ones(mgrid_coords.shape[0], 1) * times[i]

            for j in range(num_thetas):
                theta_coords = torch.ones(mgrid_coords.shape[0], 1) * thetas[j]
                theta_coords = theta_coords / (self.opt.angle_alpha * math.pi)
                coords = torch.cat((time_coords, mgrid_coords, theta_coords), dim=1)
                model_in = {'coords': coords.cuda()}
                model_out = self.model(model_in)['model_out']
                model_out = model_out.detach().cpu().numpy()
                model_out = model_out.reshape((sidelen, sidelen))

                norm_to = 0.02
                mean = 0.25
                var = 0.5
                model_out = (model_out*var/norm_to) + mean
                model_out = (model_out <= 0.001)*1.

                ax = fig.add_subplot(num_times, num_thetas, (j+1) + i*num_thetas)
                ax.set_title('t = %0.2f, theta = %0.2f' % (times[i], thetas[j]))
                s = ax.imshow(model_out.T, cmap='bwr', origin='lower', extent=(-1., 1., -1., 1.))
                fig.colorbar(s)

        fig.savefig(os.path.join(ckpt_dir, 'BRS_validation_plot_epoch_%04d.png' % epoch))


    def on_train_end(self):
        
        final_checkpoint = f"{self.model_name}_final.ckpt"
        torch.save(self.state_dict(), final_checkpoint)
        final_epoch_training_loss_text = f'Model: {self.model_name}, Final Epoch Training Loss: {self.total_train_loss_epoch[-1]}'
        final_total_training_time_text = f'Model: {self.model_name}, Final Total Training Time: {self.total_training_time[-1]}'
        self.logger.experiment.add_text('Final Epoch Training Loss', final_epoch_training_loss_text, self.current_epoch)
        self.logger.experiment.add_text('Final Total Training Time', final_total_training_time_text, self.current_epoch)
        for name, values in self.single_losses_epoch.items():
            final_loss = values[-1]
            final_loss_text = f'Model: {self.model_name}, Final {name} Loss: {final_loss}'
            self.logger.experiment.add_text(f'Final {name} Loss', final_loss_text, self.current_epoch)   


    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, model, loss_fn, lr, model_name):
        checkpoint = torch.load(checkpoint_path)
        instance = cls(model, loss_fn, lr, model_name)
        instance.load_state_dict(checkpoint)
        return instance         
