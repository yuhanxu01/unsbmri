"""Wandb logger for MRI contrast transfer training."""
import os
import time
import wandb
from .mri_visualize import visuals_to_wandb_dict


class WandbLogger:
    """Logger that uses wandb for experiment tracking and visualization."""

    def __init__(self, opt):
        """
        Initialize wandb logger.

        Args:
            opt: training options with attributes like name, checkpoints_dir, etc.
        """
        self.opt = opt
        self.name = opt.name

        # Initialize wandb
        wandb.init(
            project=getattr(opt, 'wandb_project', 'mri-contrast-transfer'),
            name=opt.name,
            config=vars(opt),
            dir=opt.checkpoints_dir,
            resume='allow',
            id=getattr(opt, 'wandb_run_id', None)
        )

        # Create logging file for losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        os.makedirs(os.path.dirname(self.log_name), exist_ok=True)
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write(f'================ Training Loss ({now}) ================\n')

        print(f'Wandb logging initialized: project={wandb.run.project}, name={wandb.run.name}, id={wandb.run.id}')

    def log_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """
        Log current losses to wandb and text file.

        Args:
            epoch: current epoch
            iters: current iteration within epoch
            losses: OrderedDict of loss names and values
            t_comp: computation time per iteration
            t_data: data loading time per iteration
        """
        # Create message for text logging
        message = f'(epoch: {epoch}, iters: {iters}, time: {t_comp:.3f}, data: {t_data:.3f}) '
        for k, v in losses.items():
            message += f'{k}: {v:.3f} '

        print(message)

        # Save to text file
        with open(self.log_name, "a") as log_file:
            log_file.write(f'{message}\n')

        # Log to wandb
        wandb_dict = {
            'epoch': epoch,
            'iteration': iters,
            'time/computation': t_comp,
            'time/data_loading': t_data
        }

        for k, v in losses.items():
            wandb_dict[f'loss/{k}'] = v

        wandb.log(wandb_dict)

    def log_current_visuals(self, visuals, epoch, step):
        """
        Log current visual results to wandb.

        Args:
            visuals: OrderedDict of visual names and tensors
            epoch: current epoch
            step: current step/iteration
        """
        mri_mode = getattr(self.opt, 'mri_representation', 'magnitude')
        images_dict = visuals_to_wandb_dict(visuals, mri_representation=mri_mode)

        # Convert numpy arrays to wandb.Image objects
        wandb_images = {}
        for label, img_array in images_dict.items():
            wandb_images[f'visuals/{label}'] = wandb.Image(img_array, caption=f'{label}_epoch{epoch}')

        wandb.log(wandb_images, step=step)

    def finish(self):
        """Finish wandb run."""
        wandb.finish()

    # Compatibility methods with old Visualizer API
    def reset(self):
        """Reset method for compatibility (no-op for wandb)."""
        pass

    def display_current_results(self, visuals, epoch, save_result):
        """
        Display current results (compatibility with old API).

        Args:
            visuals: OrderedDict of visual names and tensors
            epoch: current epoch
            save_result: whether to save results (not used, always log to wandb)
        """
        # Calculate approximate step number
        step = wandb.run.step if wandb.run else 0
        self.log_current_visuals(visuals, epoch, step)

    def plot_current_losses(self, epoch, counter_ratio, losses):
        """
        Plot current losses (compatibility with old API).
        For wandb, losses are automatically plotted when logged.

        Args:
            epoch: current epoch
            counter_ratio: progress ratio within epoch
            losses: OrderedDict of loss names and values
        """
        # Wandb automatically creates plots from logged losses
        # This method is kept for compatibility but doesn't need to do anything
        pass

    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """
        Print and log current losses (compatibility with old API).

        Args:
            epoch: current epoch
            iters: current iteration
            losses: OrderedDict of loss names and values
            t_comp: computation time
            t_data: data loading time
        """
        self.log_current_losses(epoch, iters, losses, t_comp, t_data)
