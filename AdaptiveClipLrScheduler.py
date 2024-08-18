import time
import torch
import pytorch_lightning as pl
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

class ModelLightningModule(pl.LightningModule):
    def __init__(self, model, loss_fn, lr, model_name, histogram_log_frequency=1000, max_grad_norm=1.0, weight_decay=1e-5):
        super(ModelLightningModule, self).__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.model_name = model_name
        self.histogram_log_frequency = histogram_log_frequency
        self.max_grad_norm = max_grad_norm
        self.weight_decay = weight_decay
        
        self.total_train_loss_epoch = []
        self.single_losses_epoch = {}
        self.epoch_start_time = None
        self.epoch_end_time = None
        self.total_training_time = []
        self.epoch_training_time = []
        self.single_losses_gradient_epoch = {}
        self.automatic_optimization = False
        self.gradient_histograms = {}
        self.max_grad_components = {}
        self.mean_abs_grads = {}

        # Adaptive weighting
        self.loss_weights = torch.nn.Parameter(torch.ones(len(loss_fn.loss_terms)))
        self.adaptive_weight_freq = 1000  # Update weights every 1000 steps

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5000, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "total_loss",
                "frequency": 1,
                "interval": "epoch"
            },
        }

    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()

        optimizer.zero_grad()
        model_input, gt = batch
        model_output = self(model_input)
        losses = self.loss_fn(model_output, gt)

        # Apply adaptive weighting
        weighted_losses = {name: self.loss_weights[i] * loss 
                           for i, (name, loss) in enumerate(losses.items())}
        total_loss = sum(weighted_losses.values())

        # Backward pass
        self.manual_backward(total_loss)

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)

        # Collect gradients for logging
        all_gradients = {name: [] for name in losses.keys()}
        for name in losses.keys():
            grad_dict = {}
            for param_name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_values = param.grad.detach().cpu().numpy().flatten()
                    grad_dict[param_name] = grad_values
                    all_gradients[name].extend(grad_values)
            self.single_losses_gradient_epoch[name] = grad_dict

        # Optimizer step
        optimizer.step()

        # Update adaptive weights
        if self.global_step % self.adaptive_weight_freq == 0:
            self.update_adaptive_weights(losses)

        # Logging
        self.log('total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        for name, value in weighted_losses.items():
            self.log(f'{name}_loss', value, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            if name not in self.single_losses_epoch:
                self.single_losses_epoch[name] = []
            self.single_losses_epoch[name].append(value.item())

        self.total_train_loss_epoch.append(total_loss.item())

        # Gradient logging
        for name, grads in all_gradients.items():
            if grads:
                max_grad = np.max(np.abs(grads))
                mean_abs_grad = np.mean(np.abs(grads))
            else:
                max_grad = 0
                mean_abs_grad = 0

            if name not in self.max_grad_components:
                self.max_grad_components[name] = []
            self.max_grad_components[name].append(max_grad)
            self.log(f'{name}_max_grad', max_grad, on_epoch=True, prog_bar=True, logger=True)

            if name not in self.mean_abs_grads:
                self.mean_abs_grads[name] = []
            self.mean_abs_grads[name].append(mean_abs_grad)
            self.log(f'{name}_mean_abs_grad', mean_abs_grad, on_epoch=True, prog_bar=True, logger=True)

        # Learning rate logging
        current_lr = optimizer.param_groups[0]['lr']
        self.log('learning_rate', current_lr, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Store gradients for histogram plotting
        if self.current_epoch % self.histogram_log_frequency == 0:
            self.gradient_histograms[self.current_epoch] = all_gradients

        return total_loss

    def update_adaptive_weights(self, losses):
        with torch.no_grad():
            loss_values = torch.tensor([loss.item() for loss in losses.values()])
            loss_ratios = loss_values / loss_values.mean()
            new_weights = 2 * loss_ratios / loss_ratios.sum()
            self.loss_weights.data = 0.5 * self.loss_weights + 0.5 * new_weights

    def on_train_epoch_end(self):
        self.epoch_end_time = time.time()
        epoch_training_time = self.epoch_end_time - self.epoch_start_time
        self.epoch_training_time.append(epoch_training_time)
        total_time = sum(self.epoch_training_time)
        self.total_training_time.append(total_time)

        self.log('epoch_total_loss', self.total_train_loss_epoch[-1], on_epoch=True, prog_bar=True, logger=True)
        self.log('epoch_training_time', epoch_training_time, on_epoch=True, prog_bar=True, logger=True)
        self.log('total_training_time', total_time, on_epoch=True, prog_bar=True, logger=True)

        for name, losses in self.single_losses_epoch.items():
            self.log(f'epoch_{name}_loss', np.mean(losses), on_epoch=True, prog_bar=True, logger=True)

        if self.current_epoch % 1000 == 0:
            checkpoint_filename = f"{self.model_name}_epoch_{self.current_epoch}.ckpt"
            self.save_checkpoint(checkpoint_filename)

        if self.current_epoch in self.gradient_histograms:
            self.plot_gradient_histograms(self.current_epoch)

        # Step the learning rate scheduler
        sch = self.lr_schedulers()
        sch.step(self.total_train_loss_epoch[-1])

    def save_checkpoint(self, filename):
        checkpoint_path = os.path.join(self.logger.log_dir, 'checkpoints', filename)
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(self.state_dict(), checkpoint_path)

    def on_train_end(self):
        final_checkpoint = f"{self.model_name}_final.ckpt"
        self.save_checkpoint(final_checkpoint)
        self.save_gradient_stats_to_csv()

    def save_gradient_stats_to_csv(self):
        data = {
            'epoch': range(len(self.total_train_loss_epoch)),
            'total_loss': self.total_train_loss_epoch
        }

        for name in self.single_losses_epoch.keys():
            data[f'{name}_loss'] = self.single_losses_epoch[name]
            data[f'{name}_max_grad'] = self.max_grad_components[name]
            data[f'{name}_mean_abs_grad'] = self.mean_abs_grads[name]

        df = pd.DataFrame(data)
        df.to_csv(f'{self.model_name}_gradient_stats.csv', index=False)

    def plot_gradient_histograms(self, epoch):
        gradients = self.gradient_histograms[epoch]
        fig, ax = plt.subplots(figsize=(10, 6))

        for name, grads in gradients.items():
            bins = np.linspace(-0.5, 0.5, 500)
            ax.hist(grads, bins=bins, alpha=0.5, label=name)

        ax.set_yscale('log')
        ax.set_xlim([-0.5, 0.5])
        ax.set_xlabel('Gradient Magnitude')
        ax.set_ylabel('Count')
        ax.set_title(f'Gradient Histograms at Epoch {epoch}')
        ax.legend()

        self.logger.experiment.add_figure(f'Gradient_Histograms_Epoch_{epoch}', fig, epoch)
        plt.close(fig)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, model, loss_fn, lr, model_name):
        checkpoint = torch.load(checkpoint_path)
        instance = cls(model, loss_fn, lr, model_name)
        instance.load_state_dict(checkpoint)
        return instance