# # # import torch
# # # import utils
# # # from torch.utils.tensorboard import SummaryWriter
# # # from tqdm.autonotebook import tqdm
# # # import time
# # # import numpy as np
# # # import os
# # # import shutil
# # # import matplotlib
# # # matplotlib.use('Agg')
# # # import matplotlib.pyplot as plt

# # # def plot_convergence_curves(kan_losses, mlp_losses, fourier_losses, wavelet_losses, root_path, epoch=None):
# # #     convergence_dir = os.path.join(root_path, 'convergence_curves')
    
# # #     # Ensure the convergence curves directory exists
# # #     if not os.path.exists(convergence_dir):
# # #         os.makedirs(convergence_dir)
    
# # #     plt.figure()
# # #     plt.plot(kan_losses, label='Chebyshev KAN')
# # #     plt.plot(mlp_losses, label='MLP')
# # #     plt.plot(fourier_losses, label='Fourier KAN')
# # #     plt.plot(wavelet_losses, label='Wavelet KAN')
# # #     plt.yscale('log')
# # #     plt.xlabel('Epochs')
# # #     plt.ylabel('Loss')
# # #     plt.title('Convergence Curves')
# # #     plt.legend()
# # #     if epoch is not None:
# # #         plt.savefig(os.path.join(convergence_dir, f'convergence_curves_epoch_{epoch:04d}.png'))
# # #     else:
# # #         plt.savefig(os.path.join(convergence_dir, 'convergence_curves_final.png'))
# # #     plt.close()

# # # def train(models, train_dataloader, epochs_dict, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
# # #           summary_fn=None, val_dataloader=None, double_precision=False, clip_grad=False, use_lbfgs=False, loss_schedules=None,
# # #           validation_fn=None, start_epoch=0):

# # #     optimizers = {name: torch.optim.Adam(lr=lr, params=model.parameters()) for name, model in models.items()}

# # #     if use_lbfgs:
# # #         optimizers = {name: torch.optim.LBFGS(lr=lr, params=model.parameters(), max_iter=50000, max_eval=50000,
# # #                                               history_size=50, line_search_fn='strong_wolfe') for name, model in models.items()}

# # #     if start_epoch > 0:
# # #         for name, model in models.items():
# # #             model_path = os.path.join(model_dir, f'{name}_checkpoints', 'model_epoch_%04d.pth' % start_epoch)
# # #             checkpoint = torch.load(model_path)
# # #             model.load_state_dict(checkpoint['model'])
# # #             model.train()
# # #             optimizers[name].load_state_dict(checkpoint['optimizer'])
# # #             optimizers[name].param_groups[0]['lr'] = lr
# # #             assert(start_epoch == checkpoint['epoch'])
# # #     else:
# # #         if os.path.exists(model_dir):
# # #             val = input("The model directory %s exists. Overwrite? (y/n)" % model_dir)
# # #             if val == 'y':
# # #                 shutil.rmtree(model_dir)
# # #         os.makedirs(model_dir)
# # #         for name in models.keys():
# # #             sub_dir = os.path.join(model_dir, f'{name}_model')
# # #             os.makedirs(sub_dir)
# # #             sub_checkpoints_dir = os.path.join(sub_dir, 'checkpoints')
# # #             os.makedirs(sub_checkpoints_dir)

# # #     summaries_dirs = {name: os.path.join(model_dir, f'{name}_summaries') for name in models.keys()}
# # #     for dir in summaries_dirs.values():
# # #         utils.cond_mkdir(dir)

# # #     checkpoints_dirs = {name: os.path.join(model_dir, f'{name}_checkpoints') for name in models.keys()}
# # #     for dir in checkpoints_dirs.values():
# # #         utils.cond_mkdir(dir)

# # #     writers = {name: SummaryWriter(dir) for name, dir in summaries_dirs.items()}

# # #     total_steps = 0
# # #     training_times = {name: 0 for name in models.keys()}
# # #     with tqdm(total=max(epochs_dict.values()) * len(train_dataloader)) as pbar:
# # #         train_losses = {name: [] for name in models.keys()}
# # #         for epoch in range(start_epoch, max(epochs_dict.values())):
# # #             epoch_losses = {name: 0.0 for name in models.keys()}
# # #             epoch_steps = {name: 0 for name in models.keys()}
# # #             if not epoch % epochs_til_checkpoint and epoch:
# # #                 for name, model in models.items():
# # #                     checkpoint = {
# # #                         'epoch': epoch,
# # #                         'model': model.state_dict(),
# # #                         'optimizer': optimizers[name].state_dict()}
# # #                     torch.save(checkpoint,
# # #                                os.path.join(checkpoints_dirs[name], 'model_epoch_%04d.pth' % epoch))
# # #                     np.savetxt(os.path.join(checkpoints_dirs[name], 'train_losses_epoch_%04d.txt' % epoch),
# # #                                np.array(train_losses[name]))
# # #                     if validation_fn is not None:
# # #                         validation_fn(models, checkpoints_dirs[name], epoch)
# # #                 plot_convergence_curves(train_losses['chebyshev'], train_losses['mlp'], train_losses['fourier'], train_losses['wavelet'], model_dir, epoch)
# # #             for step, (model_input, gt) in enumerate(train_dataloader):
# # #                 #print(f"Batch shape from dataloader: {model_input['coords'].shape}")
# # #                 model_input = {key: value.cuda() for key, value in model_input.items()}
# # #                 gt = {key: value.cuda() for key, value in gt.items()}

# # #                 if double_precision:
# # #                     model_input = {key: value.double() for key, value in model_input.items()}
# # #                     gt = {key: value.double() for key, value in gt.items()}

# # #                 for name, model in models.items():
# # #                     if epoch >= epochs_dict[name]:
# # #                         continue

# # #                     model_start_time = time.time()
# # #                     optimizer = optimizers[name]

# # #                     if use_lbfgs:
# # #                         def closure():
# # #                             optimizer.zero_grad()
# # #                             model_output = model(model_input)
# # #                             losses = loss_fn(model_output, gt)
# # #                             train_loss = 0.
# # #                             for loss_name, loss in losses.items():
# # #                                 train_loss += loss.mean()
# # #                             train_loss.backward()
# # #                             return train_loss
# # #                         optimizer.step(closure)
# # #                     else:
# # #                         optimizer.zero_grad()
# # #                         model_output = model(model_input)
# # #                         losses = loss_fn(model_output, gt)

# # #                         train_loss = 0.
# # #                         for loss_name, loss in losses.items():
# # #                             single_loss = loss.mean()

# # #                             if loss_schedules is not None and loss_name in loss_schedules:
# # #                                 writers[name].add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
# # #                                 single_loss *= loss_schedules[loss_name](total_steps)

# # #                             writers[name].add_scalar(loss_name, single_loss, total_steps)
# # #                             train_loss += single_loss

# # #                         epoch_losses[name] += train_loss.item()
# # #                         epoch_steps[name] += 1
# # #                         writers[name].add_scalar("total_train_loss", train_loss, total_steps)

# # #                         if not total_steps % steps_til_summary:
# # #                             torch.save(model.state_dict(),
# # #                                        os.path.join(checkpoints_dirs[name], 'model_current.pth'))

# # #                         if not use_lbfgs:
# # #                             train_loss.backward()

# # #                             if clip_grad:
# # #                                 if isinstance(clip_grad, bool):
# # #                                     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
# # #                                 else:
# # #                                     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

# # #                             optimizer.step()

# # #                     model_end_time = time.time()
# # #                     step_time = model_end_time - model_start_time
# # #                     training_times[name] += step_time
# # #                     writers[name].add_scalar("training_time", training_times[name], total_steps)

# # #                     if not total_steps % steps_til_summary:
# # #                         tqdm.write(f"Epoch {epoch}, {name} Total loss {train_loss:.6f}, "
# # #                                    f"iteration time {step_time:.6f}, "
# # #                                    f"total training time {training_times[name]:.6f}")

# # #                         if val_dataloader is not None:
# # #                             print("Running validation set...")
# # #                             model.eval()
# # #                             with torch.no_grad():
# # #                                 val_losses = []
# # #                                 for (model_input, gt) in val_dataloader:
# # #                                     model_output = model(model_input)
# # #                                     val_loss = loss_fn(model_output, gt)
# # #                                     val_losses.append(val_loss)

# # #                                 writers[name].add_scalar("val_loss", np.mean(val_losses), total_steps)
# # #                             model.train()

# # #                 total_steps += 1
# # #                 pbar.update(1)

# # #             for name in models.keys():
# # #                 if epoch_steps[name] > 0:
# # #                     train_losses[name].append(epoch_losses[name] / epoch_steps[name])
         
# # #             if epoch % epochs_til_checkpoint == 0 or epoch == max(epochs_dict.values()) - 1:
# # #                 plot_convergence_curves(train_losses['chebyshev'], train_losses['mlp'], train_losses['fourier'], train_losses['wavelet'], model_dir, epoch)

# # #         for name, model in models.items():
# # #             torch.save(model.state_dict(),
# # #                        os.path.join(checkpoints_dirs[name], 'model_final.pth'))
# # #             np.savetxt(os.path.join(checkpoints_dirs[name], 'train_losses_final.txt'),
# # #                        np.array(train_losses[name]))

# # #         plot_convergence_curves(train_losses['chebyshev'], train_losses['mlp'], train_losses['fourier'], train_losses['wavelet'], model_dir)

# # #     return train_losses['chebyshev'], train_losses['mlp'], train_losses['fourier'], train_losses['wavelet'] , training_times

# # # class LinearDecaySchedule():
# # #     def __init__(self, start_val, final_val, num_steps):
# # #         self.start_val = start_val
# # #         self.final_val = final_val
# # #         self.num_steps = num_steps

# # #     def __call__(self, iter):
# # #         return self.start_val + (self.final_val - self.start_val) * min(iter / self.num_steps, 1.)

# # import torch
# # import utils
# # from torch.utils.tensorboard import SummaryWriter
# # from tqdm.auto import tqdm
# # import time
# # import numpy as np
# # import os
# # import shutil
# # import matplotlib
# # matplotlib.use('Agg')
# # import matplotlib.pyplot as plt
# # import torch
# # import utils
# # from torch.utils.tensorboard import SummaryWriter
# # from tqdm.autonotebook import tqdm
# # import time
# # import numpy as np
# # import os
# # import shutil


# # def plot_convergence_curves(losses_dict, root_path, epoch=None):
# #     convergence_dir = os.path.join(root_path, 'convergence_curves')
    
# #     if not os.path.exists(convergence_dir):
# #         os.makedirs(convergence_dir)
    
# #     plt.figure(figsize=(10, 6))
# #     for name, losses in losses_dict.items():
# #         plt.plot(losses, label=name)
    
# #     plt.yscale('log')
# #     plt.xlabel('Iterations')
# #     plt.ylabel('Loss')
# #     plt.title('Convergence Curves')
# #     plt.legend()
    
# #     if epoch is not None:
# #         plt.savefig(os.path.join(convergence_dir, f'convergence_curves_epoch_{epoch:04d}.png'))
# #     else:
# #         plt.savefig(os.path.join(convergence_dir, 'convergence_curves_final.png'))
# #     plt.close()






# # # def plot_convergence_curves(kan_losses, mlp_losses, fourier_losses, wavelet_losses, root_path, epoch=None):
# # #     convergence_dir = os.path.join(root_path, 'convergence_curves')
    
# # #     if not os.path.exists(convergence_dir):
# # #         os.makedirs(convergence_dir)
    
# # #     plt.figure()
# # #     plt.plot(kan_losses, label='Chebyshev KAN')
# # #     plt.plot(mlp_losses, label='MLP')
# # #     plt.plot(fourier_losses, label='Fourier KAN')
# # #     plt.plot(wavelet_losses, label='Wavelet KAN')
# # #     plt.yscale('log')
# # #     plt.xlabel('Epochs')
# # #     plt.ylabel('Loss')
# # #     plt.title('Convergence Curves')
# # #     plt.legend()
# # #     if epoch is not None:
# # #         plt.savefig(os.path.join(convergence_dir, f'convergence_curves_epoch_{epoch:04d}.png'))
# # #     else:
# # #         plt.savefig(os.path.join(convergence_dir, 'convergence_curves_final.png'))
# # #     plt.close()

# # # def train(models, train_dataloader, epochs_dict, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
# # #           summary_fn=None, val_dataloader=None, double_precision=False, clip_grad=False, use_lbfgs=False, loss_schedules=None,
# # #           validation_fn=None, start_epoch=0):

# # #     assert len(models) == 1, "This function now trains one model at a time"
# # #     name, model = list(models.items())[0]

# # #     optimizer = torch.optim.Adam(lr=lr, params=model.parameters())

# # #     if use_lbfgs:
# # #         optimizer = torch.optim.LBFGS(lr=lr, params=model.parameters(), max_iter=50000, max_eval=50000,
# # #                                       history_size=50, line_search_fn='strong_wolfe')

# # #     if start_epoch > 0:
# # #         model_path = os.path.join(model_dir, f'model_epoch_{start_epoch:04d}.pth')
# # #         checkpoint = torch.load(model_path)
# # #         model.load_state_dict(checkpoint['model'])
# # #         model.train()
# # #         optimizer.load_state_dict(checkpoint['optimizer'])
# # #         optimizer.param_groups[0]['lr'] = lr
# # #         assert(start_epoch == checkpoint['epoch'])
# # #     else:
# # #         if os.path.exists(model_dir):
# # #             val = input(f"The model directory {model_dir} exists. Overwrite? (y/n)")
# # #             if val == 'y':
# # #                 shutil.rmtree(model_dir)
# # #         os.makedirs(model_dir)
# # #         os.makedirs(os.path.join(model_dir, 'checkpoints'))

# # #     summaries_dir = os.path.join(model_dir, 'summaries')
# # #     utils.cond_mkdir(summaries_dir)

# # #     checkpoints_dir = os.path.join(model_dir, 'checkpoints')
# # #     utils.cond_mkdir(checkpoints_dir)

# # #     writer = SummaryWriter(summaries_dir)

# # #     total_steps = 0
# # #     train_losses = []
# # #     iteration_times = []

# # #     for epoch in tqdm(range(start_epoch, epochs_dict[name]), desc=f"Training {name}"):
# # #         epoch_losses = []
# # #         epoch_step_times = []
        
# # #         if not epoch % epochs_til_checkpoint and epoch:
# # #             checkpoint = {
# # #                 'epoch': epoch,
# # #                 'model': model.state_dict(),
# # #                 'optimizer': optimizer.state_dict()}
# # #             torch.save(checkpoint, os.path.join(checkpoints_dir, f'model_epoch_{epoch:04d}.pth'))
# # #             np.savetxt(os.path.join(checkpoints_dir, f'train_losses_epoch_{epoch:04d}.txt'),
# # #                        np.array(train_losses))
# # #             if validation_fn is not None:
# # #                 validation_fn({name: model}, checkpoints_dir, epoch)

# # #         for step, (model_input, gt) in enumerate(tqdm(train_dataloader, desc="Steps", leave=False)):
# # #             start_time = time.time()

# # #             model_input = {key: value.cuda() for key, value in model_input.items()}
# # #             gt = {key: value.cuda() for key, value in gt.items()}

# # #             if double_precision:
# # #                 model_input = {key: value.double() for key, value in model_input.items()}
# # #                 gt = {key: value.double() for key, value in gt.items()}

# # #             if use_lbfgs:
# # #                 def closure():
# # #                     optimizer.zero_grad()
# # #                     model_output = model(model_input)
# # #                     losses = loss_fn(model_output, gt)
# # #                     train_loss = sum(loss.mean() for loss in losses.values())
# # #                     train_loss.backward()
# # #                     return train_loss
# # #                 optimizer.step(closure)
# # #             else:
# # #                 optimizer.zero_grad()
# # #                 model_output = model(model_input)
# # #                 losses = loss_fn(model_output, gt)

# # #                 train_loss = sum(loss.mean() for loss in losses.values())

# # #                 train_loss.backward()

# # #                 if clip_grad:
# # #                     if isinstance(clip_grad, bool):
# # #                         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
# # #                     else:
# # #                         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

# # #                 optimizer.step()

# # #             end_time = time.time()
# # #             step_time = end_time - start_time

# # #             epoch_losses.append(train_loss.item())
# # #             epoch_step_times.append(step_time)

# # #             if not total_steps % steps_til_summary:
# # #                 writer.add_scalar("train_loss", train_loss.item(), total_steps)
# # #                 for loss_name, loss in losses.items():
# # #                     writer.add_scalar(loss_name, loss.mean(), total_steps)

# # #             total_steps += 1

# # #         avg_loss = np.mean(epoch_losses)
# # #         avg_step_time = np.mean(epoch_step_times)
# # #         train_losses.append(avg_loss)
# # #         iteration_times.append(avg_step_time)
        
# # #         writer.add_scalar("epoch_avg_loss", avg_loss, epoch)
# # #         writer.add_scalar("epoch_avg_step_time", avg_step_time, epoch)

# # #         tqdm.write(f"Epoch {epoch}, {name}: Avg Loss {avg_loss:.6f}, "
# # #                    f"Avg Step Time {avg_step_time:.6f}")

# # #         if val_dataloader is not None and not epoch % steps_til_summary:
# # #             model.eval()
# # #             with torch.no_grad():
# # #                 val_losses = []
# # #                 for (model_input, gt) in val_dataloader:
# # #                     model_output = model(model_input)
# # #                     val_loss = loss_fn(model_output, gt)
# # #                     val_losses.append(val_loss)
# # #                 writer.add_scalar("val_loss", np.mean(val_losses), epoch)
# # #             model.train()

# # #     torch.save(model.state_dict(), os.path.join(checkpoints_dir, 'model_final.pth'))
# # #     np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'), np.array(train_losses))

# # #     return {name: train_losses}, {name: iteration_times}

# # # class LinearDecaySchedule():
# # #     def __init__(self, start_val, final_val, num_steps):
# # #         self.start_val = start_val
# # #         self.final_val = final_val
# # #         self.num_steps = num_steps

# # #     def __call__(self, iter):
# # #         return self.start_val + (self.final_val - self.start_val) * min(iter / self.num_steps, 1.)



# # def train(model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
# #           summary_fn=None, val_dataloader=None, double_precision=False, clip_grad=False, use_lbfgs=False, loss_schedules=None,
# #           validation_fn=None, start_epoch=0):

# #     optim = torch.optim.Adam(lr=lr, params=model.parameters())

# #     if use_lbfgs:
# #         optim = torch.optim.LBFGS(lr=lr, params=model.parameters(), max_iter=50000, max_eval=50000,
# #                                   history_size=50, line_search_fn='strong_wolfe')

# #     if start_epoch > 0:
# #         model_path = os.path.join(model_dir, 'checkpoints', 'model_epoch_%04d.pth' % start_epoch)
# #         checkpoint = torch.load(model_path)
# #         model.load_state_dict(checkpoint['model'])
# #         model.train()
# #         optim.load_state_dict(checkpoint['optimizer'])
# #         optim.param_groups[0]['lr'] = lr
# #         assert(start_epoch == checkpoint['epoch'])
# #     else:
# #         if os.path.exists(model_dir):
# #             val = input("The model directory %s exists. Overwrite? (y/n)"%model_dir)
# #             if val == 'y':
# #                 shutil.rmtree(model_dir)
# #         os.makedirs(model_dir)

# #     summaries_dir = os.path.join(model_dir, 'summaries')
# #     utils.cond_mkdir(summaries_dir)

# #     checkpoints_dir = os.path.join(model_dir, 'checkpoints')
# #     utils.cond_mkdir(checkpoints_dir)

# #     writer = SummaryWriter(summaries_dir)

# #     total_steps = 0
# #     with tqdm(total=len(train_dataloader) * epochs) as pbar:
# #         train_losses = []
# #         iteration_times = []
# #         for epoch in range(start_epoch, epochs):
# #             if not epoch % epochs_til_checkpoint and epoch:
# #                 checkpoint = { 
# #                     'epoch': epoch,
# #                     'model': model.state_dict(),
# #                     'optimizer': optim.state_dict()}
# #                 torch.save(checkpoint,
# #                            os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
# #                 np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),
# #                            np.array(train_losses))
# #                 if validation_fn is not None:
# #                     validation_fn(model, checkpoints_dir, epoch)

# #             for step, (model_input, gt) in enumerate(train_dataloader):
# #                 start_time = time.time()
            
# #                 model_input = {key: value.cuda() for key, value in model_input.items()}
# #                 gt = {key: value.cuda() for key, value in gt.items()}

# #                 if double_precision:
# #                     model_input = {key: value.double() for key, value in model_input.items()}
# #                     gt = {key: value.double() for key, value in gt.items()}

# #                 if use_lbfgs:
# #                     def closure():
# #                         optim.zero_grad()
# #                         model_output = model(model_input)
# #                         losses = loss_fn(model_output, gt)
# #                         train_loss = 0.
# #                         for loss_name, loss in losses.items():
# #                             train_loss += loss.mean() 
# #                         train_loss.backward()
# #                         return train_loss
# #                     optim.step(closure)

# #                 model_output = model(model_input)
# #                 losses = loss_fn(model_output, gt)

# #                 train_loss = 0.
# #                 for loss_name, loss in losses.items():
# #                     single_loss = loss.mean()

# #                     if loss_schedules is not None and loss_name in loss_schedules:
# #                         writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
# #                         single_loss *= loss_schedules[loss_name](total_steps)

# #                     writer.add_scalar(loss_name, single_loss, total_steps)
# #                     train_loss += single_loss

# #                 train_losses.append(train_loss.item())
# #                 writer.add_scalar("total_train_loss", train_loss, total_steps)

# #                 if not total_steps % steps_til_summary:
# #                     torch.save(model.state_dict(),
# #                                os.path.join(checkpoints_dir, 'model_current.pth'))

# #                 if not use_lbfgs:
# #                     optim.zero_grad()
# #                     train_loss.backward()

# #                     if clip_grad:
# #                         if isinstance(clip_grad, bool):
# #                             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
# #                         else:
# #                             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

# #                     optim.step()

# #                 iteration_time = time.time() - start_time
# #                 iteration_times.append(iteration_time)

# #                 pbar.update(1)

# #                 if not total_steps % steps_til_summary:
# #                     tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, iteration_time))

# #                     if val_dataloader is not None:
# #                         print("Running validation set...")
# #                         model.eval()
# #                         with torch.no_grad():
# #                             val_losses = []
# #                             for (model_input, gt) in val_dataloader:
# #                                 model_output = model(model_input)
# #                                 val_loss = loss_fn(model_output, gt)
# #                                 val_losses.append(val_loss)

# #                             writer.add_scalar("val_loss", np.mean(val_losses), total_steps)
# #                         model.train()

# #                 total_steps += 1

# #         torch.save(model.state_dict(),
# #                    os.path.join(checkpoints_dir, 'model_final.pth'))
# #         np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
# #                    np.array(train_losses))

# #     return train_losses, iteration_times

# # class LinearDecaySchedule():
# #     def __init__(self, start_val, final_val, num_steps):
# #         self.start_val = start_val
# #         self.final_val = final_val
# #         self.num_steps = num_steps

# #     def __call__(self, iter):
# #         return self.start_val + (self.final_val - self.start_val) * min(iter / self.num_steps, 1.)

# import torch
# import utils
# from torch.utils.tensorboard import SummaryWriter
# from tqdm.autonotebook import tqdm
# import time
# import numpy as np
# import os
# import shutil
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

# def plot_convergence_curves(kan_losses, mlp_losses, fourier_losses, wavelet_losses, root_path, epoch=None):
#     convergence_dir = os.path.join(root_path, 'convergence_curves')
    
#     # Ensure the convergence curves directory exists
#     if not os.path.exists(convergence_dir):
#         os.makedirs(convergence_dir)
    
#     plt.figure()
#     plt.plot(kan_losses, label='Chebyshev KAN')
#     plt.plot(mlp_losses, label='MLP')
#     plt.plot(fourier_losses, label='Fourier KAN')
#     plt.plot(wavelet_losses, label='Wavelet KAN')
#     plt.yscale('log')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.title('Convergence Curves')
#     plt.legend()
#     if epoch is not None:
#         plt.savefig(os.path.join(convergence_dir, f'convergence_curves_epoch_{epoch:04d}.png'))
#     else:
#         plt.savefig(os.path.join(convergence_dir, 'convergence_curves_final.png'))
#     plt.close()

# def train(models, train_dataloader, epochs_dict, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
#           summary_fn=None, val_dataloader=None, double_precision=False, clip_grad=False, use_lbfgs=False, loss_schedules=None,
#           validation_fn=None, start_epoch=0):

#     optimizers = {name: torch.optim.Adam(lr=lr, params=model.parameters()) for name, model in models.items()}

#     if use_lbfgs:
#         optimizers = {name: torch.optim.LBFGS(lr=lr, params=model.parameters(), max_iter=50000, max_eval=50000,
#                                               history_size=50, line_search_fn='strong_wolfe') for name, model in models.items()}

#     if start_epoch > 0:
#         for name, model in models.items():
#             model_path = os.path.join(model_dir, f'{name}_checkpoints', 'model_epoch_%04d.pth' % start_epoch)
#             checkpoint = torch.load(model_path)
#             model.load_state_dict(checkpoint['model'])
#             model.train()
#             optimizers[name].load_state_dict(checkpoint['optimizer'])
#             optimizers[name].param_groups[0]['lr'] = lr
#             assert(start_epoch == checkpoint['epoch'])
#     else:
#         if os.path.exists(model_dir):
#             val = input("The model directory %s exists. Overwrite? (y/n)" % model_dir)
#             if val == 'y':
#                 shutil.rmtree(model_dir)
#         os.makedirs(model_dir)
#         for name in models.keys():
#             sub_dir = os.path.join(model_dir, f'{name}_model')
#             os.makedirs(sub_dir)
#             sub_checkpoints_dir = os.path.join(sub_dir, 'checkpoints')
#             os.makedirs(sub_checkpoints_dir)

#     summaries_dirs = {name: os.path.join(model_dir, f'{name}_summaries') for name in models.keys()}
#     for dir in summaries_dirs.values():
#         utils.cond_mkdir(dir)

#     checkpoints_dirs = {name: os.path.join(model_dir, f'{name}_checkpoints') for name in models.keys()}
#     for dir in checkpoints_dirs.values():
#         utils.cond_mkdir(dir)

#     writers = {name: SummaryWriter(dir) for name, dir in summaries_dirs.items()}

#     total_steps = 0
#     training_times = {name: 0 for name in models.keys()}
#     with tqdm(total=max(epochs_dict.values()) * len(train_dataloader)) as pbar:
#         train_losses = {name: [] for name in models.keys()}
#         for epoch in range(start_epoch, max(epochs_dict.values())):
#             epoch_losses = {name: 0.0 for name in models.keys()}
#             epoch_steps = {name: 0 for name in models.keys()}
#             if not epoch % epochs_til_checkpoint and epoch:
#                 for name, model in models.items():
#                     checkpoint = {
#                         'epoch': epoch,
#                         'model': model.state_dict(),
#                         'optimizer': optimizers[name].state_dict()}
#                     torch.save(checkpoint,
#                                os.path.join(checkpoints_dirs[name], 'model_epoch_%04d.pth' % epoch))
#                     np.savetxt(os.path.join(checkpoints_dirs[name], 'train_losses_epoch_%04d.txt' % epoch),
#                                np.array(train_losses[name]))
#                     if validation_fn is not None:
#                         validation_fn(models, checkpoints_dirs[name], epoch)
#                 plot_convergence_curves(train_losses['chebyshev'], train_losses['mlp'], train_losses['fourier'], train_losses['wavelet'], model_dir, epoch)
#             for step, (model_input, gt) in enumerate(train_dataloader):
#                 start_time = time.time()
#                 model_input = {key: value.cuda() for key, value in model_input.items()}
#                 gt = {key: value.cuda() for key, value in gt.items()}

#                 if double_precision:
#                     model_input = {key: value.double() for key, value in model_input.items()}
#                     gt = {key: value.double() for key, value in gt.items()}

#                 for name, model in models.items():
#                     if epoch >= epochs_dict[name]:
#                         continue

#                     optimizer = optimizers[name]
#                     model_start_time = start_time

#                     if use_lbfgs:
#                         def closure():
#                             optimizer.zero_grad()
#                             model_output = model(model_input)
#                             losses = loss_fn(model_output, gt)
#                             train_loss = 0.
#                             for loss_name, loss in losses.items():
#                                 train_loss += loss.mean()
#                             train_loss.backward()
#                             return train_loss
#                         optimizer.step(closure)
#                     else:
#                         optimizer.zero_grad()
#                         model_output = model(model_input)
#                         losses = loss_fn(model_output, gt)

#                         train_loss = 0.
#                         for loss_name, loss in losses.items():
#                             single_loss = loss.mean()

#                             if loss_schedules is not None and loss_name in loss_schedules:
#                                 writers[name].add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
#                                 single_loss *= loss_schedules[loss_name](total_steps)

#                             writers[name].add_scalar(loss_name, single_loss, total_steps)
#                             train_loss += single_loss

#                         epoch_losses[name] += train_loss.item()
#                         epoch_steps[name] += 1
#                         writers[name].add_scalar("total_train_loss", train_loss, total_steps)
#                         if not use_lbfgs:
#                             train_loss.backward()

#                             if clip_grad:
#                                 if isinstance(clip_grad, bool):
#                                     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
#                                 else:
#                                     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

#                                 optimizer.step()
#                         pbar.update(1)

#                     model_end_time = time.time()
#                     step_time = model_end_time - model_start_time
#                     training_times[name] += step_time
                           
#                     if not total_steps % steps_til_summary:
#                         torch.save(model.state_dict(),
#                                        os.path.join(checkpoints_dirs[name], 'model_current.pth'))

#                         tqdm.write(f"Epoch {epoch}, {name} Total loss {train_loss:.6f}, "
#                                    f"iteration time {step_time:.6f}, "
#                                    f"total training time {training_times[name]:.6f}")               

#                         if val_dataloader is not None:
#                             print("Running validation set...")
#                             model.eval()
#                             with torch.no_grad():
#                                 val_losses = []
#                                 for (model_input, gt) in val_dataloader:
#                                     model_output = model(model_input)
#                                     val_loss = loss_fn(model_output, gt)
#                                     val_losses.append(val_loss)

#                                 writers[name].add_scalar("val_loss", np.mean(val_losses), total_steps)
#                             model.train()
                            
#                     writers[name].add_scalar("training_time", training_times[name], total_steps)
        

#                     total_steps += 1
                    


#             for name in models.keys():
#                 if epoch_steps[name] > 0:
#                     train_losses[name].append(epoch_losses[name] / epoch_steps[name])
         
#             if epoch % epochs_til_checkpoint == 0 or epoch == max(epochs_dict.values()) - 1:
#                 plot_convergence_curves(train_losses['chebyshev'], train_losses['mlp'], train_losses['fourier'], train_losses['wavelet'], model_dir, epoch)

#         for name, model in models.items():
#             torch.save(model.state_dict(),
#                        os.path.join(checkpoints_dirs[name], 'model_final.pth'))
#             np.savetxt(os.path.join(checkpoints_dirs[name], 'train_losses_final.txt'),
#                        np.array(train_losses[name]))

#         plot_convergence_curves(train_losses['chebyshev'], train_losses['mlp'], train_losses['fourier'], train_losses['wavelet'], model_dir)

#     return train_losses['chebyshev'], train_losses['mlp'], train_losses['fourier'], train_losses['wavelet'], training_times

# class LinearDecaySchedule():
#     def __init__(self, start_val, final_val, num_steps):
#         self.start_val = start_val
#         self.final_val = final_val
#         self.num_steps = num_steps

#     def __call__(self, iter):
#         return self.start_val + (self.final_val - self.start_val) * min(iter / self.num_steps, 1.)


# import torch
# import utils
# from torch.utils.tensorboard import SummaryWriter
# from tqdm.autonotebook import tqdm
# import time
# import numpy as np
# import os
# import shutil
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

# def plot_convergence_curves(kan_losses, mlp_losses, fourier_losses, wavelet_losses, root_path, epoch=None):
#     convergence_dir = os.path.join(root_path, 'convergence_curves')
    
#     # Ensure the convergence curves directory exists
#     if not os.path.exists(convergence_dir):
#         os.makedirs(convergence_dir)
    
#     plt.figure()
#     plt.plot(kan_losses, label='Chebyshev KAN')
#     plt.plot(mlp_losses, label='MLP')
#     plt.plot(fourier_losses, label='Fourier KAN')
#     plt.plot(wavelet_losses, label='Wavelet KAN')
#     plt.yscale('log')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.title('Convergence Curves')
#     plt.legend()
#     if epoch is not None:
#         plt.savefig(os.path.join(convergence_dir, f'convergence_curves_epoch_{epoch:04d}.png'))
#     else:
#         plt.savefig(os.path.join(convergence_dir, 'convergence_curves_final.png'))
#     plt.close()

# def train(models, train_dataloader, epochs_dict, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
#           summary_fn=None, val_dataloader=None, double_precision=False, clip_grad=False, use_lbfgs=False, loss_schedules=None,
#           validation_fn=None, start_epoch=0):

#     optimizers = {name: torch.optim.Adam(lr=lr, params=model.parameters()) for name, model in models.items()}

#     if use_lbfgs:
#         optimizers = {name: torch.optim.LBFGS(lr=lr, params=model.parameters(), max_iter=50000, max_eval=50000,
#                                               history_size=50, line_search_fn='strong_wolfe') for name, model in models.items()}

#     if start_epoch > 0:
#         for name, model in models.items():
#             model_path = os.path.join(model_dir, f'{name}_checkpoints', 'model_epoch_%04d.pth' % start_epoch)
#             checkpoint = torch.load(model_path)
#             model.load_state_dict(checkpoint['model'])
#             model.train()
#             optimizers[name].load_state_dict(checkpoint['optimizer'])
#             optimizers[name].param_groups[0]['lr'] = lr
#             assert(start_epoch == checkpoint['epoch'])
#     else:
#         if os.path.exists(model_dir):
#             val = input("The model directory %s exists. Overwrite? (y/n)" % model_dir)
#             if val == 'y':
#                 shutil.rmtree(model_dir)
#         os.makedirs(model_dir)
#         for name in models.keys():
#             sub_dir = os.path.join(model_dir, f'{name}_model')
#             os.makedirs(sub_dir)
#             sub_checkpoints_dir = os.path.join(sub_dir, 'checkpoints')
#             os.makedirs(sub_checkpoints_dir)

#     summaries_dirs = {name: os.path.join(model_dir, f'{name}_summaries') for name in models.keys()}
#     for dir in summaries_dirs.values():
#         utils.cond_mkdir(dir)

#     checkpoints_dirs = {name: os.path.join(model_dir, f'{name}_checkpoints') for name in models.keys()}
#     for dir in checkpoints_dirs.values():
#         utils.cond_mkdir(dir)

#     writers = {name: SummaryWriter(dir) for name, dir in summaries_dirs.items()}

#     total_steps = 0
#     training_times = {name: 0 for name in models.keys()}
#     with tqdm(total=max(epochs_dict.values()) * len(train_dataloader)) as pbar:
#         train_losses = {name: [] for name in models.keys()}
#         for epoch in range(start_epoch, max(epochs_dict.values())):
#             epoch_losses = {name: 0.0 for name in models.keys()}
#             epoch_steps = {name: 0 for name in models.keys()}
#             if not epoch % epochs_til_checkpoint and epoch:
#                 for name, model in models.items():
#                     checkpoint = {
#                         'epoch': epoch,
#                         'model': model.state_dict(),
#                         'optimizer': optimizers[name].state_dict()}
#                     torch.save(checkpoint,
#                                os.path.join(checkpoints_dirs[name], 'model_epoch_%04d.pth' % epoch))
#                     np.savetxt(os.path.join(checkpoints_dirs[name], 'train_losses_epoch_%04d.txt' % epoch),
#                                np.array(train_losses[name]))
#                     if validation_fn is not None:
#                         validation_fn(models, checkpoints_dirs[name], epoch)
#                 plot_convergence_curves(train_losses['chebyshev'], train_losses['mlp'], train_losses['fourier'], train_losses['wavelet'], model_dir, epoch)
#             for step, (model_input, gt) in enumerate(train_dataloader):
#                 start_time = time.time()
#                 model_input = {key: value.cuda() for key, value in model_input.items()}
#                 gt = {key: value.cuda() for key, value in gt.items()}

#                 if double_precision:
#                     model_input = {key: value.double() for key, value in model_input.items()}
#                     gt = {key: value.double() for key, value in gt.items()}

#                 for name, model in models.items():
#                     if epoch >= epochs_dict[name]:
#                         continue

#                     optimizer = optimizers[name]
#                     model_start_time = start_time

#                     if use_lbfgs:
#                         def closure():
#                             optimizer.zero_grad()
#                             model_output = model(model_input)
#                             losses = loss_fn(model_output, gt)
#                             train_loss = 0.
#                             for loss_name, loss in losses.items():
#                                 train_loss += loss.mean()
#                             train_loss.backward()
#                             return train_loss
#                         optimizer.step(closure)
#                     else:
#                         optimizer.zero_grad()
#                         model_output = model(model_input)
#                         losses = loss_fn(model_output, gt)

#                         train_loss = 0.
#                         for loss_name, loss in losses.items():
#                             single_loss = loss.mean()

#                             if loss_schedules is not None and loss_name in loss_schedules:
#                                 writers[name].add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
#                                 single_loss *= loss_schedules[loss_name](total_steps)

#                             writers[name].add_scalar(loss_name, single_loss, total_steps)
#                             train_loss += single_loss

#                         epoch_losses[name] += train_loss.item()
#                         epoch_steps[name] += 1
#                         writers[name].add_scalar("total_train_loss", train_loss, total_steps)

#                         if not total_steps % steps_til_summary:
#                             torch.save(model.state_dict(),
#                                        os.path.join(checkpoints_dirs[name], 'model_current.pth'))

#                         if not use_lbfgs:
#                             #optimizer.zero_grad()
#                             train_loss.backward()

#                             if clip_grad:
#                                 if isinstance(clip_grad, bool):
#                                     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
#                                 else:
#                                     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

#                             optimizer.step()
#                         pbar.update(1)

#                         model_end_time = time.time()
#                         step_time = model_end_time - model_start_time
#                         training_times[name] += step_time
#                         if not total_steps % steps_til_summary:
#                             tqdm.write(f"Epoch {epoch}, {name} Total loss {train_loss:.6f}, "
#                                        f"iteration time {step_time:.6f}, "
#                                        f"total training time {training_times[name]:.6f}")               

#                             if val_dataloader is not None:
#                                 print("Running validation set...")
#                                 model.eval()
#                                 with torch.no_grad():
#                                     val_losses = []
#                                     for (model_input, gt) in val_dataloader:
#                                         model_output = model(model_input)
#                                         val_loss = loss_fn(model_output, gt)
#                                         val_losses.append(val_loss)

#                                     writers[name].add_scalar("val_loss", np.mean(val_losses), total_steps)
#                                 model.train()
                            
#                         writers[name].add_scalar("training_time", training_times[name], total_steps)
        

#                         total_steps += 1
                    


#             for name in models.keys():
#                 if epoch_steps[name] > 0:
#                     train_losses[name].append(epoch_losses[name] / epoch_steps[name])
         
#             if epoch % epochs_til_checkpoint == 0 or epoch == max(epochs_dict.values()) - 1:
#                 plot_convergence_curves(train_losses['chebyshev'], train_losses['mlp'], train_losses['fourier'], train_losses['wavelet'], model_dir, epoch)

#         for name, model in models.items():
#             torch.save(model.state_dict(),
#                        os.path.join(checkpoints_dirs[name], 'model_final.pth'))
#             np.savetxt(os.path.join(checkpoints_dirs[name], 'train_losses_final.txt'),
#                        np.array(train_losses[name]))

#         plot_convergence_curves(train_losses['chebyshev'], train_losses['mlp'], train_losses['fourier'], train_losses['wavelet'], model_dir)

#     return train_losses['chebyshev'], train_losses['mlp'], train_losses['fourier'], train_losses['wavelet'], training_times

# class LinearDecaySchedule():
#     def __init__(self, start_val, final_val, num_steps):
#         self.start_val = start_val
#         self.final_val = final_val
#         self.num_steps = num_steps

#     def __call__(self, iter):
#         return self.start_val + (self.final_val - self.start_val) * min(iter / self.num_steps, 1.)


# import torch
# import utils
# from torch.utils.tensorboard import SummaryWriter
# from tqdm.autonotebook import tqdm
# import time
# import numpy as np
# import os
# import shutil
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

# def plot_convergence_curves(kan_losses, mlp_losses, fourier_losses, wavelet_losses, root_path, epoch=None):
#     convergence_dir = os.path.join(root_path, 'convergence_curves')
    
#     # Ensure the convergence curves directory exists
#     if not os.path.exists(convergence_dir):
#         os.makedirs(convergence_dir)
    
#     plt.figure()
#     plt.plot(kan_losses, label='Chebyshev KAN')
#     plt.plot(mlp_losses, label='MLP')
#     plt.plot(fourier_losses, label='Fourier KAN')
#     plt.plot(wavelet_losses, label='Wavelet KAN')
#     plt.yscale('log')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.title('Convergence Curves')
#     plt.legend()
#     if epoch is not None:
#         plt.savefig(os.path.join(convergence_dir, f'convergence_curves_epoch_{epoch:04d}.png'))
#     else:
#         plt.savefig(os.path.join(convergence_dir, 'convergence_curves_final.png'))
#     plt.close()

# def train(models, train_dataloader, epochs_dict, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
#           summary_fn=None, val_dataloader=None, double_precision=False, clip_grad=False, use_lbfgs=False, loss_schedules=None,
#           validation_fn=None, start_epoch=0):

#     optimizers = {name: torch.optim.Adam(lr=lr, params=model.parameters()) for name, model in models.items()}

#     if use_lbfgs:
#         optimizers = {name: torch.optim.LBFGS(lr=lr, params=model.parameters(), max_iter=50000, max_eval=50000,
#                                               history_size=50, line_search_fn='strong_wolfe') for name, model in models.items()}

#     if start_epoch > 0:
#         for name, model in models.items():
#             model_path = os.path.join(model_dir, f'{name}_checkpoints', 'model_epoch_%04d.pth' % start_epoch)
#             checkpoint = torch.load(model_path)
#             model.load_state_dict(checkpoint['model'])
#             model.train()
#             optimizers[name].load_state_dict(checkpoint['optimizer'])
#             optimizers[name].param_groups[0]['lr'] = lr
#             assert(start_epoch == checkpoint['epoch'])
#     else:
#         if os.path.exists(model_dir):
#             val = input("The model directory %s exists. Overwrite? (y/n)" % model_dir)
#             if val == 'y':
#                 shutil.rmtree(model_dir)
#         os.makedirs(model_dir)
#         for name in models.keys():
#             sub_dir = os.path.join(model_dir, f'{name}_model')
#             os.makedirs(sub_dir)
#             sub_checkpoints_dir = os.path.join(sub_dir, 'checkpoints')
#             os.makedirs(sub_checkpoints_dir)

#     summaries_dirs = {name: os.path.join(model_dir, f'{name}_summaries') for name in models.keys()}
#     for dir in summaries_dirs.values():
#         utils.cond_mkdir(dir)

#     checkpoints_dirs = {name: os.path.join(model_dir, f'{name}_checkpoints') for name in models.keys()}
#     for dir in checkpoints_dirs.values():
#         utils.cond_mkdir(dir)

#     writers = {name: SummaryWriter(dir) for name, dir in summaries_dirs.items()}

#     total_steps = 0
#     training_times = {name: 0 for name in models.keys()}
#     with tqdm(total=max(epochs_dict.values()) * len(train_dataloader)) as pbar:
#         train_losses = {name: [] for name in models.keys()}
#         for epoch in range(start_epoch, max(epochs_dict.values())):
#             epoch_losses = {name: 0.0 for name in models.keys()}
#             epoch_steps = {name: 0 for name in models.keys()}
#             if not epoch % epochs_til_checkpoint and epoch:
#                 for name, model in models.items():
#                     checkpoint = {
#                         'epoch': epoch,
#                         'model': model.state_dict(),
#                         'optimizer': optimizers[name].state_dict()}
#                     torch.save(checkpoint,
#                                os.path.join(checkpoints_dirs[name], 'model_epoch_%04d.pth' % epoch))
#                     np.savetxt(os.path.join(checkpoints_dirs[name], 'train_losses_epoch_%04d.txt' % epoch),
#                                np.array(train_losses[name]))
#                     if validation_fn is not None:
#                         validation_fn(models, checkpoints_dirs[name], epoch)
#                 plot_convergence_curves(train_losses['chebyshev'], train_losses['mlp'], train_losses['fourier'], train_losses['wavelet'], model_dir, epoch)
            
#             for step, (model_input, gt) in enumerate(train_dataloader):
#                 start_time = time.time()
#                 model_input = {key: value.cuda() for key, value in model_input.items()}
#                 gt = {key: value.cuda() for key, value in gt.items()}

#                 if double_precision:
#                     model_input = {key: value.double() for key, value in model_input.items()}
#                     gt = {key: value.double() for key, value in gt.items()}

#                 for name, model in models.items():
#                     if epoch >= epochs_dict[name]:
#                         continue

#                     optimizer = optimizers[name]
#                     model_start_time = start_time

#                     if use_lbfgs:
#                         def closure():
#                             optimizer.zero_grad()
#                             model_output = model(model_input)
#                             losses = loss_fn(model_output, gt)
#                             train_loss = 0.
#                             for loss_name, loss in losses.items():
#                                 train_loss += loss.mean()
#                             train_loss.backward()
#                             return train_loss
#                         optimizer.step(closure)
#                     else:
#                         optimizer.zero_grad()
#                         model_output = model(model_input)
#                         losses = loss_fn(model_output, gt)

#                         train_loss = 0.
#                         for loss_name, loss in losses.items():
#                             single_loss = loss.mean()

#                             if loss_schedules is not None and loss_name in loss_schedules:
#                                 writers[name].add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
#                                 single_loss *= loss_schedules[loss_name](total_steps)

#                             writers[name].add_scalar(loss_name, single_loss, total_steps)
#                             train_loss += single_loss

#                         epoch_losses[name] += train_loss.item()
#                         epoch_steps[name] += 1
#                         writers[name].add_scalar("total_train_loss", train_loss, total_steps)

#                         train_loss.backward()

#                         if clip_grad:
#                             if isinstance(clip_grad, bool):
#                                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
#                             else:
#                                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

#                         optimizer.step()

#                     model_end_time = time.time()
#                     step_time = model_end_time - model_start_time
#                     training_times[name] += step_time
#                     writers[name].add_scalar("training_time", training_times[name], total_steps)

#                     total_steps += 1

#                 pbar.update(1)

#                 if not total_steps % steps_til_summary:
#                     for name in models.keys():
#                         if epoch < epochs_dict[name]:
#                             tqdm.write(f"Epoch {epoch}, {name} Total loss {epoch_losses[name]/epoch_steps[name]:.6f}, "
#                                        f"iteration time {step_time:.6f}, "
#                                        f"total training time {training_times[name]:.6f}")

#                     if val_dataloader is not None:
#                         print("Running validation set...")
#                         for name, model in models.items():
#                             if epoch < epochs_dict[name]:
#                                 model.eval()
#                                 with torch.no_grad():
#                                     val_losses = []
#                                     for (model_input, gt) in val_dataloader:
#                                         model_output = model(model_input)
#                                         val_loss = loss_fn(model_output, gt)
#                                         val_losses.append(val_loss)

#                                     writers[name].add_scalar("val_loss", np.mean(val_losses), total_steps)
#                                 model.train()

#             for name in models.keys():
#                 if epoch_steps[name] > 0:
#                     train_losses[name].append(epoch_losses[name] / epoch_steps[name])

#             if epoch % epochs_til_checkpoint == 0 or epoch == max(epochs_dict.values()) - 1:
#                 plot_convergence_curves(train_losses['chebyshev'], train_losses['mlp'], train_losses['fourier'], train_losses['wavelet'], model_dir, epoch)

#         for name, model in models.items():
#             torch.save(model.state_dict(),
#                        os.path.join(checkpoints_dirs[name], 'model_final.pth'))
#             np.savetxt(os.path.join(checkpoints_dirs[name], 'train_losses_final.txt'),
#                        np.array(train_losses[name]))

#         plot_convergence_curves(train_losses['chebyshev'], train_losses['mlp'], train_losses['fourier'], train_losses['wavelet'], model_dir)

#     return train_losses['chebyshev'], train_losses['mlp'], train_losses['fourier'], train_losses['wavelet'], training_times

# class LinearDecaySchedule():
#     def __init__(self, start_val, final_val, num_steps):
#         self.start_val = start_val
#         self.final_val = final_val
#         self.num_steps = num_steps

#     def __call__(self, iter):
#         return self.start_val + (self.final_val - self.start_val) * min(iter / self.num_steps, 1.)



# import torch
# import utils
# from torch.utils.tensorboard import SummaryWriter
# from tqdm.autonotebook import tqdm
# import time
# import numpy as np
# import os
# import shutil
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

# def plot_convergence_curves(kan_losses, mlp_losses, fourier_losses, wavelet_losses, root_path, epoch=None):
#     convergence_dir = os.path.join(root_path, 'convergence_curves')
    
#     if not os.path.exists(convergence_dir):
#         os.makedirs(convergence_dir)
    
#     plt.figure()
#     plt.plot(kan_losses, label='Chebyshev KAN')
#     plt.plot(mlp_losses, label='MLP')
#     plt.plot(fourier_losses, label='Fourier KAN')
#     plt.plot(wavelet_losses, label='Wavelet KAN')
#     plt.yscale('log')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.title('Convergence Curves')
#     plt.legend()
#     if epoch is not None:
#         plt.savefig(os.path.join(convergence_dir, f'convergence_curves_epoch_{epoch:04d}.png'))
#     else:
#         plt.savefig(os.path.join(convergence_dir, 'convergence_curves_final.png'))
#     plt.close()

# def train(models, train_dataloader, epochs_dict, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
#           summary_fn=None, val_dataloader=None, double_precision=False, clip_grad=False, use_lbfgs=False, loss_schedules=None,
#           validation_fn=None, start_epoch=0):

#     optimizers = {name: torch.optim.Adam(lr=lr, params=model.parameters()) for name, model in models.items()}

#     if use_lbfgs:
#         optimizers = {name: torch.optim.LBFGS(lr=lr, params=model.parameters(), max_iter=50000, max_eval=50000,
#                                               history_size=50, line_search_fn='strong_wolfe') for name, model in models.items()}

#     if start_epoch > 0:
#         for name, model in models.items():
#             model_path = os.path.join(model_dir, f'{name}_checkpoints', 'model_epoch_%04d.pth' % start_epoch)
#             checkpoint = torch.load(model_path)
#             model.load_state_dict(checkpoint['model'])
#             model.train()
#             optimizers[name].load_state_dict(checkpoint['optimizer'])
#             optimizers[name].param_groups[0]['lr'] = lr
#             assert(start_epoch == checkpoint['epoch'])
#     else:
#         if os.path.exists(model_dir):
#             val = input("The model directory %s exists. Overwrite? (y/n)" % model_dir)
#             if val == 'y':
#                 shutil.rmtree(model_dir)
#         os.makedirs(model_dir)
#         for name in models.keys():
#             sub_dir = os.path.join(model_dir, f'{name}_model')
#             os.makedirs(sub_dir)
#             sub_checkpoints_dir = os.path.join(sub_dir, 'checkpoints')
#             os.makedirs(sub_checkpoints_dir)

#     summaries_dirs = {name: os.path.join(model_dir, f'{name}_summaries') for name in models.keys()}
#     for dir in summaries_dirs.values():
#         utils.cond_mkdir(dir)

#     checkpoints_dirs = {name: os.path.join(model_dir, f'{name}_checkpoints') for name in models.keys()}
#     for dir in checkpoints_dirs.values():
#         utils.cond_mkdir(dir)

#     writers = {name: SummaryWriter(dir) for name, dir in summaries_dirs.items()}

#     total_steps = 0
#     training_times = {name: 0 for name in models.keys()}
#     with tqdm(total=max(epochs_dict.values()) * len(train_dataloader)) as pbar:
#         train_losses = {name: [] for name in models.keys()}
#         for epoch in range(start_epoch, max(epochs_dict.values())):
#             epoch_losses = {name: 0.0 for name in models.keys()}
#             epoch_steps = {name: 0 for name in models.keys()}
#             if not epoch % epochs_til_checkpoint and epoch:
#                 for name, model in models.items():
#                     checkpoint = {
#                         'epoch': epoch,
#                         'model': model.state_dict(),
#                         'optimizer': optimizers[name].state_dict()}
#                     torch.save(checkpoint,
#                                os.path.join(checkpoints_dirs[name], 'model_epoch_%04d.pth' % epoch))
#                     np.savetxt(os.path.join(checkpoints_dirs[name], 'train_losses_epoch_%04d.txt' % epoch),
#                                np.array(train_losses[name]))
#                     if validation_fn is not None:
#                         validation_fn(models, checkpoints_dirs[name], epoch)
#                 plot_convergence_curves(train_losses['chebyshev'], train_losses['mlp'], train_losses['fourier'], train_losses['wavelet'], model_dir, epoch)
            
#             for step, (model_input, gt) in enumerate(train_dataloader):
#                 model_input = {key: value.cuda() for key, value in model_input.items()}
#                 gt = {key: value.cuda() for key, value in gt.items()}

#                 if double_precision:
#                     model_input = {key: value.double() for key, value in model_input.items()}
#                     gt = {key: value.double() for key, value in gt.items()}

#                 for name, model in models.items():
#                     if epoch >= epochs_dict[name]:
#                         continue

#                     model_start_time = time.time()

#                     optimizer = optimizers[name]

#                     if use_lbfgs:
#                         def closure():
#                             optimizer.zero_grad()
#                             model_output = model(model_input)
#                             losses = loss_fn(model_output, gt)
#                             train_loss = 0.
#                             for loss_name, loss in losses.items():
#                                 train_loss += loss.mean()
#                             train_loss.backward()
#                             return train_loss
#                         optimizer.step(closure)
#                     else:
#                         optimizer.zero_grad()
#                         model_output = model(model_input)
#                         losses = loss_fn(model_output, gt)

#                         train_loss = 0.
#                         for loss_name, loss in losses.items():
#                             single_loss = loss.mean()

#                             if loss_schedules is not None and loss_name in loss_schedules:
#                                 writers[name].add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
#                                 single_loss *= loss_schedules[loss_name](total_steps)

#                             writers[name].add_scalar(loss_name, single_loss, total_steps)
#                             train_loss += single_loss

#                         epoch_losses[name] += train_loss.item()
#                         epoch_steps[name] += 1
#                         writers[name].add_scalar("total_train_loss", train_loss, total_steps)

#                         train_loss.backward()

#                         if clip_grad:
#                             if isinstance(clip_grad, bool):
#                                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
#                             else:
#                                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

#                         optimizer.step()

#                     model_end_time = time.time()
#                     step_time = model_end_time - model_start_time
#                     training_times[name] += step_time
#                     writers[name].add_scalar("training_time", training_times[name], total_steps)

#                     if not total_steps % steps_til_summary:
#                         tqdm.write(f"Epoch {epoch}, {name} Total loss {train_loss:.6f}, "
#                                    f"iteration time {step_time:.6f}, "
#                                    f"total training time {training_times[name]:.6f}")

#                     total_steps += 1

#                 pbar.update(1)

#                 if not total_steps % steps_til_summary:
#                     if val_dataloader is not None:
#                         print("Running validation set...")
#                         for name, model in models.items():
#                             if epoch < epochs_dict[name]:
#                                 model.eval()
#                                 with torch.no_grad():
#                                     val_losses = []
#                                     for (val_model_input, val_gt) in val_dataloader:
#                                         val_model_output = model(val_model_input)
#                                         val_loss = loss_fn(val_model_output, val_gt)
#                                         val_losses.append(val_loss)

#                                     writers[name].add_scalar("val_loss", np.mean(val_losses), total_steps)
#                                 model.train()

#             for name in models.keys():
#                 if epoch_steps[name] > 0:
#                     train_losses[name].append(epoch_losses[name] / epoch_steps[name])

#             if epoch % epochs_til_checkpoint == 0 or epoch == max(epochs_dict.values()) - 1:
#                 plot_convergence_curves(train_losses['chebyshev'], train_losses['mlp'], train_losses['fourier'], train_losses['wavelet'], model_dir, epoch)

#         for name, model in models.items():
#             torch.save(model.state_dict(),
#                        os.path.join(checkpoints_dirs[name], 'model_final.pth'))
#             np.savetxt(os.path.join(checkpoints_dirs[name], 'train_losses_final.txt'),
#                        np.array(train_losses[name]))

#         plot_convergence_curves(train_losses['chebyshev'], train_losses['mlp'], train_losses['fourier'], train_losses['wavelet'], model_dir)

#     return train_losses['chebyshev'], train_losses['mlp'], train_losses['fourier'], train_losses['wavelet'], training_times

# class LinearDecaySchedule():
#     def __init__(self, start_val, final_val, num_steps):
#         self.start_val = start_val
#         self.final_val = final_val
#         self.num_steps = num_steps

#     def __call__(self, iter):
#         return self.start_val + (self.final_val - self.start_val) * min(iter / self.num_steps, 1.)


# import torch
# import utils
# from torch.utils.tensorboard import SummaryWriter
# from tqdm.autonotebook import tqdm
# import time
# import numpy as np
# import os
# import shutil
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

# def plot_convergence_curves(kan_losses, mlp_losses, fourier_losses, wavelet_losses, root_path, epoch=None):
#     convergence_dir = os.path.join(root_path, 'convergence_curves')
    
#     if not os.path.exists(convergence_dir):
#         os.makedirs(convergence_dir)
    
#     plt.figure()
#     plt.plot(kan_losses, label='Chebyshev KAN')
#     plt.plot(mlp_losses, label='MLP')
#     plt.plot(fourier_losses, label='Fourier KAN')
#     plt.plot(wavelet_losses, label='Wavelet KAN')
#     plt.yscale('log')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.title('Convergence Curves')
#     plt.legend()
#     if epoch is not None:
#         plt.savefig(os.path.join(convergence_dir, f'convergence_curves_epoch_{epoch:04d}.png'))
#     else:
#         plt.savefig(os.path.join(convergence_dir, 'convergence_curves_final.png'))
#     plt.close()

# def train(models, train_dataloader, epochs_dict, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
#           summary_fn=None, val_dataloader=None, double_precision=False, clip_grad=False, use_lbfgs=False, loss_schedules=None,
#           validation_fn=None, start_epoch=0):

#     optimizers = {name: torch.optim.Adam(lr=lr, params=model.parameters()) for name, model in models.items()}

#     if use_lbfgs:
#         optimizers = {name: torch.optim.LBFGS(lr=lr, params=model.parameters(), max_iter=50000, max_eval=50000,
#                                               history_size=50, line_search_fn='strong_wolfe') for name, model in models.items()}

#     if start_epoch > 0:
#         for name, model in models.items():
#             model_path = os.path.join(model_dir, f'{name}_checkpoints', 'model_epoch_%04d.pth' % start_epoch)
#             checkpoint = torch.load(model_path)
#             model.load_state_dict(checkpoint['model'])
#             model.train()
#             optimizers[name].load_state_dict(checkpoint['optimizer'])
#             optimizers[name].param_groups[0]['lr'] = lr
#             assert(start_epoch == checkpoint['epoch'])
#     else:
#         if os.path.exists(model_dir):
#             val = input("The model directory %s exists. Overwrite? (y/n)" % model_dir)
#             if val == 'y':
#                 shutil.rmtree(model_dir)
#         os.makedirs(model_dir)
#         for name in models.keys():
#             sub_dir = os.path.join(model_dir, f'{name}_model')
#             os.makedirs(sub_dir)
#             sub_checkpoints_dir = os.path.join(sub_dir, 'checkpoints')
#             os.makedirs(sub_checkpoints_dir)

#     summaries_dirs = {name: os.path.join(model_dir, f'{name}_summaries') for name in models.keys()}
#     for dir in summaries_dirs.values():
#         utils.cond_mkdir(dir)

#     checkpoints_dirs = {name: os.path.join(model_dir, f'{name}_checkpoints') for name in models.keys()}
#     for dir in checkpoints_dirs.values():
#         utils.cond_mkdir(dir)

#     writers = {name: SummaryWriter(dir) for name, dir in summaries_dirs.items()}

#     total_steps = 0
#     training_times = {name: 0 for name in models.keys()}
#     with tqdm(total=max(epochs_dict.values()) * len(train_dataloader)) as pbar:
#         train_losses = {name: [] for name in models.keys()}
#         for epoch in range(start_epoch, max(epochs_dict.values())):
#             epoch_losses = {name: 0.0 for name in models.keys()}
#             epoch_steps = {name: 0 for name in models.keys()}
#             if not epoch % epochs_til_checkpoint and epoch:
#                 for name, model in models.items():
#                     checkpoint = {
#                         'epoch': epoch,
#                         'model': model.state_dict(),
#                         'optimizer': optimizers[name].state_dict()}
#                     torch.save(checkpoint,
#                                os.path.join(checkpoints_dirs[name], 'model_epoch_%04d.pth' % epoch))
#                     np.savetxt(os.path.join(checkpoints_dirs[name], 'train_losses_epoch_%04d.txt' % epoch),
#                                np.array(train_losses[name]))
#                     if validation_fn is not None:
#                         validation_fn(models, checkpoints_dirs[name], epoch)
#                 plot_convergence_curves(train_losses['chebyshev'], train_losses['mlp'], train_losses['fourier'], train_losses['wavelet'], model_dir, epoch)
            
#             for step, (model_input, gt) in enumerate(train_dataloader):
#                 model_input = {key: value.cuda() for key, value in model_input.items()}
#                 gt = {key: value.cuda() for key, value in gt.items()}

#                 if double_precision:
#                     model_input = {key: value.double() for key, value in model_input.items()}
#                     gt = {key: value.double() for key, value in gt.items()}

#                 for name, model in models.items():
#                     if epoch >= epochs_dict[name]:
#                         continue

#                     model_start_time = time.time()

#                     optimizer = optimizers[name]

#                     if use_lbfgs:
#                         def closure():
#                             optimizer.zero_grad()
#                             model_output = model(model_input)
#                             losses = loss_fn(model_output, gt)
#                             train_loss = 0.
#                             for loss_name, loss in losses.items():
#                                 train_loss += loss.mean()
#                             train_loss.backward()
#                             return train_loss
#                         optimizer.step(closure)
#                     else:
#                         optimizer.zero_grad()
#                         model_output = model(model_input)
#                         losses = loss_fn(model_output, gt)

#                         train_loss = 0.
#                         for loss_name, loss in losses.items():
#                             single_loss = loss.mean()

#                             if loss_schedules is not None and loss_name in loss_schedules:
#                                 writers[name].add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
#                                 single_loss *= loss_schedules[loss_name](total_steps)

#                             writers[name].add_scalar(loss_name, single_loss, total_steps)
#                             train_loss += single_loss

#                         epoch_losses[name] += train_loss.item()
#                         epoch_steps[name] += 1
#                         writers[name].add_scalar("total_train_loss", train_loss, total_steps)

#                         train_loss.backward()

#                         if clip_grad:
#                             if isinstance(clip_grad, bool):
#                                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
#                             else:
#                                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

#                         optimizer.step()

#                     model_end_time = time.time()
#                     step_time = model_end_time - model_start_time
#                     training_times[name] += step_time
#                     writers[name].add_scalar("training_time", training_times[name], total_steps)

#                     total_steps += 1

#                 pbar.update(1)

#             # Print losses for all models every 100 epochs
#             if epoch % 100 == 0:
#                 for name in models.keys():
#                     if epoch < epochs_dict[name]:
#                         avg_loss = epoch_losses[name] / epoch_steps[name] if epoch_steps[name] > 0 else 0
#                         tqdm.write(f"Epoch {epoch}, {name} Total loss {avg_loss:.6f}, "
#                                    f"total training time {training_times[name]:.6f}")

#                 if val_dataloader is not None:
#                     print("Running validation set...")
#                     for name, model in models.items():
#                         if epoch < epochs_dict[name]:
#                             model.eval()
#                             with torch.no_grad():
#                                 val_losses = []
#                                 for (val_model_input, val_gt) in val_dataloader:
#                                     val_model_output = model(val_model_input)
#                                     val_loss = loss_fn(val_model_output, val_gt)
#                                     val_losses.append(val_loss)

#                                 writers[name].add_scalar("val_loss", np.mean(val_losses), total_steps)
#                             model.train()

#             for name in models.keys():
#                 if epoch_steps[name] > 0:
#                     train_losses[name].append(epoch_losses[name] / epoch_steps[name])

#             if epoch % epochs_til_checkpoint == 0 or epoch == max(epochs_dict.values()) - 1:
#                 plot_convergence_curves(train_losses['chebyshev'], train_losses['mlp'], train_losses['fourier'], train_losses['wavelet'], model_dir, epoch)

#         for name, model in models.items():
#             torch.save(model.state_dict(),
#                        os.path.join(checkpoints_dirs[name], 'model_final.pth'))
#             np.savetxt(os.path.join(checkpoints_dirs[name], 'train_losses_final.txt'),
#                        np.array(train_losses[name]))

#         plot_convergence_curves(train_losses['chebyshev'], train_losses['mlp'], train_losses['fourier'], train_losses['wavelet'], model_dir)

#     return train_losses['chebyshev'], train_losses['mlp'], train_losses['fourier'], train_losses['wavelet'], training_times

# class LinearDecaySchedule():
#     def __init__(self, start_val, final_val, num_steps):
#         self.start_val = start_val
#         self.final_val = final_val
#         self.num_steps = num_steps

#     def __call__(self, iter):
#         return self.start_val + (self.final_val - self.start_val) * min(iter / self.num_steps, 1.)



import torch
import utils
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import time
import numpy as np
import os
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_convergence_curves(kan_losses, mlp_losses, fourier_losses, wavelet_losses, root_path, epoch=None):
    convergence_dir = os.path.join(root_path, 'convergence_curves')
    
    if not os.path.exists(convergence_dir):
        os.makedirs(convergence_dir)
    
    plt.figure()
    plt.plot(kan_losses, label='Chebyshev KAN')
    plt.plot(mlp_losses, label='MLP')
    plt.plot(fourier_losses, label='Fourier KAN')
    plt.plot(wavelet_losses, label='Wavelet KAN')
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Convergence Curves')
    plt.legend()
    if epoch is not None:
        plt.savefig(os.path.join(convergence_dir, f'convergence_curves_epoch_{epoch:04d}.png'))
    else:
        plt.savefig(os.path.join(convergence_dir, 'convergence_curves_final.png'))
    plt.close()

def train(models, train_dataloader, epochs_dict, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
          summary_fn=None, val_dataloader=None, double_precision=False, clip_grad=False, use_lbfgs=False, loss_schedules=None,
          validation_fn=None, start_epoch=0):

    optimizers = {name: torch.optim.Adam(lr=lr, params=model.parameters()) for name, model in models.items()}

    if use_lbfgs:
        optimizers = {name: torch.optim.LBFGS(lr=lr, params=model.parameters(), max_iter=50000, max_eval=50000,
                                              history_size=50, line_search_fn='strong_wolfe') for name, model in models.items()}

    if start_epoch > 0:
        for name, model in models.items():
            model_path = os.path.join(model_dir, f'{name}_checkpoints', 'model_epoch_%04d.pth' % start_epoch)
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model'])
            model.train()
            optimizers[name].load_state_dict(checkpoint['optimizer'])
            optimizers[name].param_groups[0]['lr'] = lr
            assert(start_epoch == checkpoint['epoch'])
    else:
        if os.path.exists(model_dir):
            val = input("The model directory %s exists. Overwrite? (y/n)" % model_dir)
            if val == 'y':
                shutil.rmtree(model_dir)
        os.makedirs(model_dir)
        for name in models.keys():
            sub_dir = os.path.join(model_dir, f'{name}_model')
            os.makedirs(sub_dir)
            sub_checkpoints_dir = os.path.join(sub_dir, 'checkpoints')
            os.makedirs(sub_checkpoints_dir)

    summaries_dirs = {name: os.path.join(model_dir, f'{name}_summaries') for name in models.keys()}
    for dir in summaries_dirs.values():
        utils.cond_mkdir(dir)

    checkpoints_dirs = {name: os.path.join(model_dir, f'{name}_checkpoints') for name in models.keys()}
    for dir in checkpoints_dirs.values():
        utils.cond_mkdir(dir)

    writers = {name: SummaryWriter(dir) for name, dir in summaries_dirs.items()}

    total_steps = 0
    training_times = {name: 0 for name in models.keys()}
    with tqdm(total=max(epochs_dict.values()) * len(train_dataloader)) as pbar:
        train_losses = {name: [] for name in models.keys()}
        for epoch in range(start_epoch, max(epochs_dict.values())):
            epoch_losses = {name: 0.0 for name in models.keys()}
            epoch_steps = {name: 0 for name in models.keys()}
            epoch_times = {name: 0.0 for name in models.keys()}  # Reset timing for each epoch
            
            if not epoch % epochs_til_checkpoint and epoch:
                for name, model in models.items():
                    checkpoint = {
                        'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizers[name].state_dict()}
                    torch.save(checkpoint,
                               os.path.join(checkpoints_dirs[name], 'model_epoch_%04d.pth' % epoch))
                    np.savetxt(os.path.join(checkpoints_dirs[name], 'train_losses_epoch_%04d.txt' % epoch),
                               np.array(train_losses[name]))
                    if validation_fn is not None:
                        validation_fn(models, checkpoints_dirs[name], epoch)
                plot_convergence_curves(train_losses['chebyshev'], train_losses['mlp'], train_losses['fourier'], train_losses['wavelet'], model_dir, epoch)
            
            for step, (model_input, gt) in enumerate(train_dataloader):
                model_input = {key: value.cuda() for key, value in model_input.items()}
                gt = {key: value.cuda() for key, value in gt.items()}

                if double_precision:
                    model_input = {key: value.double() for key, value in model_input.items()}
                    gt = {key: value.double() for key, value in gt.items()}

                for name, model in models.items():
                    if epoch >= epochs_dict[name]:
                        continue

                    model_start_time = time.time()

                    optimizer = optimizers[name]

                    if use_lbfgs:
                        def closure():
                            optimizer.zero_grad()
                            model_output = model(model_input)
                            losses = loss_fn(model_output, gt)
                            train_loss = 0.
                            for loss_name, loss in losses.items():
                                train_loss += loss.mean()
                            train_loss.backward()
                            return train_loss
                        optimizer.step(closure)
                    else:
                        optimizer.zero_grad()
                        model_output = model(model_input)
                        losses = loss_fn(model_output, gt)

                        train_loss = 0.
                        for loss_name, loss in losses.items():
                            single_loss = loss.mean()

                            if loss_schedules is not None and loss_name in loss_schedules:
                                writers[name].add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
                                single_loss *= loss_schedules[loss_name](total_steps)

                            writers[name].add_scalar(loss_name, single_loss, total_steps)
                            train_loss += single_loss

                        epoch_losses[name] += train_loss.item()
                        epoch_steps[name] += 1
                        writers[name].add_scalar("total_train_loss", train_loss, total_steps)

                        train_loss.backward()

                        if clip_grad:
                            if isinstance(clip_grad, bool):
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                            else:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                        optimizer.step()

                    model_end_time = time.time()
                    step_time = model_end_time - model_start_time
                    epoch_times[name] += step_time
                    training_times[name] += step_time
                    
                    # Log iteration time and total training time to TensorBoard
                    writers[name].add_scalar("iteration_time", step_time, total_steps)
                    writers[name].add_scalar("total_training_time", training_times[name], total_steps)

                    total_steps += 1

                pbar.update(1)

            # Print losses for all models every 100 epochs
            if epoch % 100 == 0:
                for name in models.keys():
                    if epoch < epochs_dict[name]:
                        avg_loss = epoch_losses[name] / epoch_steps[name] if epoch_steps[name] > 0 else 0
                        avg_iteration_time = epoch_times[name] / epoch_steps[name] if epoch_steps[name] > 0 else 0
                        tqdm.write(f"Epoch {epoch}, {name} Total loss {avg_loss:.6f}, "
                                   f"avg iteration time {avg_iteration_time:.6f}, "
                                   f"epoch time {epoch_times[name]:.6f}, "
                                   f"total training time {training_times[name]:.6f}")

                if val_dataloader is not None:
                    print("Running validation set...")
                    for name, model in models.items():
                        if epoch < epochs_dict[name]:
                            model.eval()
                            with torch.no_grad():
                                val_losses = []
                                for (val_model_input, val_gt) in val_dataloader:
                                    val_model_output = model(val_model_input)
                                    val_loss = loss_fn(val_model_output, val_gt)
                                    val_losses.append(val_loss)

                                writers[name].add_scalar("val_loss", np.mean(val_losses), total_steps)
                            model.train()

            for name in models.keys():
                if epoch_steps[name] > 0:
                    train_losses[name].append(epoch_losses[name] / epoch_steps[name])

            if epoch % epochs_til_checkpoint == 0 or epoch == max(epochs_dict.values()) - 1:
                plot_convergence_curves(train_losses['chebyshev'], train_losses['mlp'], train_losses['fourier'], train_losses['wavelet'], model_dir, epoch)

        for name, model in models.items():
            torch.save(model.state_dict(),
                       os.path.join(checkpoints_dirs[name], 'model_final.pth'))
            np.savetxt(os.path.join(checkpoints_dirs[name], 'train_losses_final.txt'),
                       np.array(train_losses[name]))

        plot_convergence_curves(train_losses['chebyshev'], train_losses['mlp'], train_losses['fourier'], train_losses['wavelet'], model_dir)

    return train_losses['chebyshev'], train_losses['mlp'], train_losses['fourier'], train_losses['wavelet'], training_times

class LinearDecaySchedule():
    def __init__(self, start_val, final_val, num_steps):
        self.start_val = start_val
        self.final_val = final_val
        self.num_steps = num_steps

    def __call__(self, iter):
        return self.start_val + (self.final_val - self.start_val) * min(iter / self.num_steps, 1.)