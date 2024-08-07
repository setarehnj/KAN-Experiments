# Train the models

# chebyshev_lightning = ModelLightningModule(chebyshev_model, loss_fn, opt.lr, opt.steps_til_summary, val_fn)
# dnn_lightning = ModelLightningModule(dnn_model, loss_fn, opt.lr, opt.steps_til_summary, val_fn)
# fourier_lightning = ModelLightningModule(fourier_model, loss_fn, opt.lr, opt.steps_til_summary, val_fn)
# wavelet_lightning = ModelLightningModule(wavelet_model, loss_fn, opt.lr, opt.steps_til_summary, val_fn)

# # Define the trainers
# # Define the trainers
# trainer_chebyshev = pl.Trainer(max_epochs=opt.chebyshev_num_epochs, accelerator='gpu', devices=[0], log_every_n_steps=opt.steps_til_summary)
# trainer_dnn = pl.Trainer(max_epochs=opt.dnn_num_epochs, accelerator='gpu', devices=[0], log_every_n_steps=opt.steps_til_summary)
# trainer_fourier = pl.Trainer(max_epochs=opt.fourier_num_epochs, accelerator='gpu', devices=[1], log_every_n_steps=opt.steps_til_summary)
# trainer_wavelet = pl.Trainer(max_epochs=opt.wavelet_num_epochs, accelerator='gpu', devices=[1], log_every_n_steps=opt.steps_til_summary)

# # Train the models

# trainer_chebyshev.fit(chebyshev_lightning, dataloader)
# trainer_dnn.fit(dnn_lightning, dataloader)
# trainer_fourier.fit(fourier_lightning, dataloader)
# trainer_wavelet.fit(wavelet_lightning, dataloader)

# # Plot convergence curves

# plot_convergence_curves(chebyshev_lightning.losses, dnn_lightning.losses, fourier_lightning.losses, wavelet_lightning.losses, root_path)

# # Ensure dataset consistency

# for model_name, model in models.items():
#     for i, (model_input, gt) in enumerate(dataloader):
#         print(f"Model: {model_name}, Batch: {i}, Input: {model_input}, GT: {gt}")
#         if i >= 1:  # Print only the first two batches
#             break





  # def training_step(self, batch, batch_idx):
    #     core-start_time = time.time()
    #     model_input, gt = batch
    #     model_output = self(model_input)
    #     losses = self.loss_fn(model_output, gt)
    #     train_loss = sum([loss.mean() for loss in losses.values()])
    #     self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    #     self.losses.append(train_loss.item())
    #     return train_loss

    # def on_train_epoch_end(self, outputs):
    #     iteration_time = time.time() - self.start_time  # Calculate the epoch time
    #     self.iteration_times.append(iteration_time)
    #     self.total_training_times.append(sum(self.iteration_times))

    #     self.log('iteration_time', iteration_time, on_epoch=True, prog_bar=True, logger=True)
    #     self.log('total_training_time', self.total_training_times[-1], on_epoch=True, prog_bar=True, logger=True)

    #     if self.current_epoch % self.steps_til_summary == 0:
    #         torch.save(self.model.state_dict(), f"{self.model.__class__.__name__}_epoch_{self.current_epoch}.ckpt")
