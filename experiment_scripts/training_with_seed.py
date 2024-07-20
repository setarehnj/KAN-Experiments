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
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def plot_convergence_curves(losses, training_times, root_path):
    convergence_dir = os.path.join(root_path, 'convergence_curves')
    
    if not os.path.exists(convergence_dir):
        os.makedirs(convergence_dir)
    
    plt.figure(figsize=(10, 5))
    for name, loss in losses.items():
        plt.plot(loss, label=name)
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Convergence Curves')
    plt.legend()
    plt.savefig(os.path.join(convergence_dir, 'convergence_curves_final.png'))
    plt.close()

    plt.figure(figsize=(10, 5))
    names = list(training_times.keys())
    times = list(training_times.values())
    plt.bar(names, times)
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Times for Each Model')
    plt.savefig(os.path.join(convergence_dir, 'training_times.png'))
    plt.close()

def train_single_model(model, model_name, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, 
                       model_dir, loss_fn, summary_fn=None, val_dataloader=None, double_precision=False, 
                       clip_grad=False, use_lbfgs=False, loss_schedules=None, validation_fn=None, start_epoch=0):
    
    optimizer = torch.optim.Adam(lr=lr, params=model.parameters())

    if use_lbfgs:
        optimizer = torch.optim.LBFGS(lr=lr, params=model.parameters(), max_iter=50000, max_eval=50000,
                                      history_size=50, line_search_fn='strong_wolfe')

    model_dir = os.path.join(model_dir, f'{model_name}_model')
    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    summaries_dir = os.path.join(model_dir, 'summaries')

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(summaries_dir, exist_ok=True)

    writer = SummaryWriter(summaries_dir)

    total_steps = 0
    training_time = 0
    with tqdm(total=epochs * len(train_dataloader), desc=f"Training {model_name}") as pbar:
        train_losses = []
        for epoch in range(start_epoch, epochs):
            epoch_loss = 0.0
            epoch_steps = 0
            
            for step, (model_input, gt) in enumerate(train_dataloader):
                step_start_time = time.time()

                model_input = {key: value.cuda() for key, value in model_input.items()}
                gt = {key: value.cuda() for key, value in gt.items()}

                if double_precision:
                    model_input = {key: value.double() for key, value in model_input.items()}
                    gt = {key: value.double() for key, value in gt.items()}

                if use_lbfgs:
                    def closure():
                        optimizer.zero_grad()
                        model_output = model(model_input)
                        losses = loss_fn(model_output, gt)
                        train_loss = sum(loss.mean() for loss in losses.values())
                        train_loss.backward()
                        return train_loss
                    optimizer.step(closure)
                else:
                    optimizer.zero_grad()
                    model_output = model(model_input)
                    losses = loss_fn(model_output, gt)

                    train_loss = sum(loss.mean() for loss in losses.values())

                    if loss_schedules is not None:
                        for loss_name, loss_schedule in loss_schedules.items():
                            writer.add_scalar(f"{loss_name}_weight", loss_schedule(total_steps), total_steps)
                            train_loss *= loss_schedule(total_steps)

                    writer.add_scalar("train_loss", train_loss, total_steps)
                    epoch_loss += train_loss.item()
                    epoch_steps += 1

                    train_loss.backward()

                    if clip_grad:
                        if isinstance(clip_grad, bool):
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                    optimizer.step()

                step_time = time.time() - step_start_time
                training_time += step_time
                writer.add_scalar("iteration_time", step_time, total_steps)
                writer.add_scalar("total_training_time", training_time, total_steps)

                total_steps += 1
                pbar.update(1)

            avg_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0
            train_losses.append(avg_loss)

            if epoch % 100 == 0:
                tqdm.write(f"Epoch {epoch}, {model_name} Total loss {avg_loss:.6f}, "
                           f"total training time {training_time:.6f}")

            if epoch % epochs_til_checkpoint == 0 or epoch == epochs - 1:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }, os.path.join(checkpoints_dir, f'model_epoch_{epoch:04d}.pth'))

        torch.save(model.state_dict(), os.path.join(checkpoints_dir, 'model_final.pth'))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'), np.array(train_losses))

    return train_losses, training_time

def train_all_models(models, train_dataloader, epochs_dict, lr, steps_til_summary, epochs_til_checkpoint, 
                     model_dir, loss_fn, summary_fn=None, val_dataloader=None, double_precision=False, 
                     clip_grad=False, use_lbfgs=False, loss_schedules=None, validation_fn=None, start_epoch=0, seed=42):
    
    set_seed(seed)

    all_losses = {}
    all_training_times = {}

    for name, model in models.items():
        print(f"Training {name} model")
        train_dataloader.dataset.reset()  # Reset the dataset for each model
        losses, training_time = train_single_model(
            model, name, train_dataloader, epochs_dict[name], lr, steps_til_summary, epochs_til_checkpoint,
            model_dir, loss_fn, summary_fn, val_dataloader, double_precision, clip_grad, use_lbfgs,
            loss_schedules, validation_fn, start_epoch
        )
        all_losses[name] = losses
        all_training_times[name] = training_time

    plot_convergence_curves(all_losses, all_training_times, model_dir)

    return all_losses, all_training_times

# In your main script:
losses, training_times = train_all_models(
    models=models, train_dataloader=dataloader, epochs_dict=epochs_dict, lr=opt.lr,
    steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
    model_dir=root_path, loss_fn=loss_fn, clip_grad=opt.clip_grad,
    use_lbfgs=opt.use_lbfgs, validation_fn=val_fn, seed=opt.seed
)

# Print final training times
for model_name, time in training_times.items():
    print(f"Training time for {model_name}: {time:.2f} seconds")