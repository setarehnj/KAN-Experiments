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

def plot_convergence_curves(cheby_losses, mlp_losses, root_path, epoch=None):
    convergence_dir = os.path.join(root_path, 'convergence_curves')
    
    # Ensure the convergence curves directory exists
    if not os.path.exists(convergence_dir):
        os.makedirs(convergence_dir)
    
    plt.figure()
    plt.plot(cheby_losses, label='KAN')
    plt.plot(mlp_losses, label='MLP')
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

    optimizers = {name: torch.optim.Adam(lr=lr, params=model.parameters()) for name, model in models.items()}  # line added: create optimizers for each model

    if use_lbfgs:
        optimizers = {name: torch.optim.LBFGS(lr=lr, params=model.parameters(), max_iter=50000, max_eval=50000,
                                              history_size=50, line_search_fn='strong_wolfe') for name, model in models.items()}  # line added: create LBFGS optimizers for each model

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
        for name in models.keys():  # line added: create directories for each model
            sub_dir = os.path.join(model_dir, f'{name}_model')
            os.makedirs(sub_dir)
            sub_checkpoints_dir = os.path.join(sub_dir, 'checkpoints')
            os.makedirs(sub_checkpoints_dir)

    summaries_dirs = {name: os.path.join(model_dir, f'{name}_summaries') for name in models.keys()}  # line added: create summaries directories for each model
    for dir in summaries_dirs.values():
        utils.cond_mkdir(dir)

    checkpoints_dirs = {name: os.path.join(model_dir, f'{name}_checkpoints') for name in models.keys()}  # line added: create checkpoints directories for each model
    for dir in checkpoints_dirs.values():
        utils.cond_mkdir(dir)

    writers = {name: SummaryWriter(dir) for name, dir in summaries_dirs.items()}  # line added: create writers for each model

    total_steps = 0
    with tqdm(total=max(epochs_dict.values()) * len(train_dataloader)) as pbar:
        train_losses = {name: [] for name in models.keys()}  # line added: create train_losses for each model
        for epoch in range(start_epoch, max(epochs_dict.values())):
            epoch_losses = {name: 0.0 for name in models.keys()}  # Initialize epoch losses
            epoch_steps = {name: 0 for name in models.keys()}  
            if not epoch % epochs_til_checkpoint and epoch:
                for name, model in models.items():  # line added: save checkpoints for each model
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
                plot_convergence_curves(train_losses['kan'], train_losses['mlp'], model_dir, epoch)
            for step, (model_input, gt) in enumerate(train_dataloader):
                start_time = time.time()

                model_input = {key: value.cuda() for key, value in model_input.items()}
                gt = {key: value.cuda() for key, value in gt.items()}

                if double_precision:
                    model_input = {key: value.double() for key, value in model_input.items()}
                    gt = {key: value.double() for key, value in gt.items()}

                for name, model in models.items():  # line added: iterate through each model
                   # print(name)
                    optimizer = optimizers[name]
                   # print(optimizer)

                    if epoch >= epochs_dict[name]:
                        continue

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

                       # train_losses[name].append(train_loss.item())
                        epoch_losses[name] += train_loss.item()  # Accumulate losses for this epoch
                        epoch_steps[name] += 1  # Increment step count for this model
                        writers[name].add_scalar("total_train_loss", train_loss, total_steps)

                        if not total_steps % steps_til_summary:
                            torch.save(model.state_dict(),
                                       os.path.join(checkpoints_dirs[name], 'model_current.pth'))

                        if not use_lbfgs:
                            train_loss.backward()

                            if clip_grad:
                                if isinstance(clip_grad, bool):
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                                else:
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                            optimizer.step()

                        if not total_steps % steps_til_summary:
                            tqdm.write("Epoch %d, %s Total loss %0.6f, iteration time %0.6f" % (epoch, name, train_loss, time.time() - start_time))

                            if val_dataloader is not None:
                                print("Running validation set...")
                                model.eval()
                                with torch.no_grad():
                                    val_losses = []
                                    for (model_input, gt) in val_dataloader:
                                        model_output = model(model_input)
                                        val_loss = loss_fn(model_output, gt)
                                        val_losses.append(val_loss)

                                    writers[name].add_scalar("val_loss", np.mean(val_losses), total_steps)
                                model.train()

                total_steps += 1
                pbar.update(1)

            for name in models.keys():
                if epoch_steps[name] > 0:
                    train_losses[name].append(epoch_losses[name] / epoch_steps[name])
         
            if epoch % epochs_til_checkpoint == 0 or epoch == max(epochs_dict.values()) - 1:
                plot_convergence_curves(train_losses['kan'], train_losses['mlp'], model_dir, epoch)

        for name, model in models.items():  # line added: save final state for each model
            torch.save(model.state_dict(),
                       os.path.join(checkpoints_dirs[name], 'model_final.pth'))
            np.savetxt(os.path.join(checkpoints_dirs[name], 'train_losses_final.txt'),
                       np.array(train_losses[name]))

        plot_convergence_curves(train_losses['kan'], train_losses['mlp'], model_dir)

    return train_losses['kan'], train_losses['mlp']
class LinearDecaySchedule():
    def __init__(self, start_val, final_val, num_steps):
        self.start_val = start_val
        self.final_val = final_val
        self.num_steps = num_steps

    def __call__(self, iter):
        return self.start_val + (self.final_val - self.start_val) * min(iter / self.num_steps, 1.)


