"""Test script for MRI contrast transfer.

Runs inference on test data and saves results to disk.
Results are saved to --results_dir organized by image type.
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
import util.util as util
from util.mri_visualize import visuals_to_wandb_dict
from PIL import Image


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True

    dataset = create_dataset(opt)
    dataset2 = create_dataset(opt)
    train_dataset = create_dataset(util.copyconf(opt, phase="train"))
    model = create_model(opt)

    # Create results directory
    save_dir = os.path.join(opt.results_dir, opt.name, f'{opt.phase}_{opt.epoch}')
    os.makedirs(save_dir, exist_ok=True)
    print(f'Saving results to {save_dir}')

    for i, (data, data2) in enumerate(zip(dataset, dataset2)):
        if i == 0:
            model.data_dependent_initialize(data, data2)
            model.setup(opt)
            model.parallelize()
            if opt.eval:
                model.eval()
        if i >= opt.num_test:
            break

        model.set_input(data, data2)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()

        if i % 5 == 0:
            print(f'Processing ({i:04d})-th image... {img_path}')

        # Convert visuals to images
        mri_mode = getattr(opt, 'mri_representation', 'magnitude')
        images_dict = visuals_to_wandb_dict(visuals, mri_representation=mri_mode)

        # Save images
        base_name = os.path.splitext(os.path.basename(img_path[0]))[0]
        for label, img_array in images_dict.items():
            label_dir = os.path.join(save_dir, label)
            os.makedirs(label_dir, exist_ok=True)
            save_path = os.path.join(label_dir, f'{base_name}.png')
            Image.fromarray(img_array).save(save_path)

    print(f'Test complete. Results saved to {save_dir}')
