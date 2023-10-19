# 2D 2.5D 3D Segmentation Project
# Aneja Lab | Yale School of Medicine
# Developed by Arman Avesta, MD
# Created (3/30/22)
# Updated (11/5/22)

# -------------------------------------------------- Imports --------------------------------------------------

# Project imports:

from data_loader import AdniDataset, make_image_list
from capsnet_model_3d import CapsNet3D
from loss_functions import DiceLoss
from glob import glob
from data_set_uwmadison_3D import DatasetUWMadison3D
import albumentations as Album

# System imports:

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import os
from os.path import join
from shutil import copyfile
from datetime import datetime
from tqdm import tqdm, trange
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dipy.io.image import save_nifti


class CFG:
    # seed = 101
    # debug         = False # set debug=False for Full Training
    # exp_name      = '2.5D'
    # comment       = 'unet-efficientnet_b0-160x192-ep=5'
    # model_name    = 'Unet'
    # backbone      = 'efficientnet-b0'
    # train_batch_size = 15
    # valid_batch_size = 15
    test_batch_size = 2
    # img_size = [64, 64]  # [160, 192]

    # epochs        = 5
    # lr            = 2e-3
    # scheduler     = 'CosineAnnealingLR'
    # min_lr        = 1e-6
    # T_max         = int(30000/train_bs*epochs)+50
    # T_0           = 25
    # warmup_epochs = 0
    # wd            = 1e-6
    # n_accumulate  = max(1, 32//train_bs)
    n_fold = 5
    # folds         = [0]
    # num_classes   = 3
    # device        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    project_root = '/mnt/d/code_medimg_aneja_lab'
    
    # Backup S3 destination, if not None:
    # s3_results_folder = 's3://aneja-lab-capsnet/data/results/temp'
    ec2_results_folder = "test_results_" + datetime.now().strftime("%y%m%d_%H%M%S")
    s3_results_folder = None


data_transforms = {
    "test": Album.Compose([
        ], p=1.0),
}


# ---------------------------------------------- TestUNet3D class ----------------------------------------------

class TestCapsNet3D:

    def __init__(self, saved_model_path=None):
        self.start_time = datetime.now()

        ##########################################################
        #                  SET TESTING PARAMETERS                #
        ##########################################################

        # Set segmentation target:
        # self.output_structure = 'right hippocampus'
        # Set FreeSurfer code for segmentation target:
        # to find the code, open any aparc+aseg.mgz in FreeView and change color coding to lookup table
        # self.output_code = 53

        # Set the size of the cropped volume:
        # if this is set to 100, the center of the volumed is cropped with the size of 100 x 100 x 100.
        # if this is set to (100, 64, 64), the center of the volume is cropped with size of (100 x 64 x 64).
        # note that 100, 64 and 64 here respectively represent left-right, posterior-anterior,
        # and inferior-superior dimensions, i.e. standard radiology coordinate system ('L','A','S').
        # self.crop = (64, 64, 64)
        # Set cropshift:
        # if the target structure is right hippocampus, the crop box may be shifted to right by 20 pixels,
        # anterior by 5 pixels, and inferior by 20 pixels --> cropshift = (-20, 5, -20);
        # note that crop and cropshift here are set here using standard radiology system ('L','A','S'):
        # self.cropshift = (-20, 0, -20)

        # Set loss function: options are DiceLoss, criterion, and IoULoss:
        self.criterion = DiceLoss(conversion='threshold', reduction='none')

        # Project root:
        # CFG.project_root = '/mnt/d/code_medimg_aneja_lab'

        # Saved model paths:
        self.saved_model_folder = 'train_results_231017_171404'  # ZONA - the code here should take parameter of model file/folder OR "latest" which will search for latest file
        self.saved_model_filename = 'saved_model.pth.tar'

        # Testing dataset paths:
        self.datasets_folder = 'data_uwmadison_01c_preprocessed_3d'  # ZONA - a) the preprocessing script needs to establish TEST data
        # Testing on validation or test set:
        # self.set = 'validation set'
        # csv file containing list of inputs for testing:
        # self.test_inputs_csv = 'valid_inputs.csv'
        # csv file containing list of outputs for testing:
        # self.test_outputs_csv = 'valid_outputs.csv'
        # csv file to which testing losses will be saved:

        # Set batch size (upper limit is determined by GPU memory):
        # CFG.test_batch_size = 16

        # Set model: UNet3D
        self.model = CapsNet3D()

        # .......................................................................................................

        # Folder to save results and nifti files:
        # (cropped inputs, model predictions, and ground truth together with individual scan losses)
        # self.niftis_folder = 'data/results/temp/niftis'
        # csv file to which testing hyperparameters and results will be saved:
        # self.hyperparameters_file = 'testing_hyperparameters.csv'

        # Determine if backup to S3 should be done:
        # self.s3backup = True
        # S3 bucket backup folder for results:
        # CFG.s3_results_folder = 's3://aneja-lab-capsnet/data/results/temp/niftis'

        # .......................................................................................................
        ###################################
        #   DON'T CHANGE THESE, PLEASE!   #
        ###################################

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load model:
        self.saved_model_path = (join(CFG.project_root, self.saved_model_folder, self.saved_model_filename)
                                 if saved_model_path is None else saved_model_path)
        self.load_model()

        # # Load testing dataset:
        # self.inputs_paths = make_image_list(join(CFG.project_root, self.datasets_folder,
        #                                          self.test_inputs_csv))
        # self.outputs_paths = make_image_list(join(CFG.project_root, self.datasets_folder,
        #                                           self.test_outputs_csv))
        # self.dataset = AdniDataset(self.inputs_paths, self.outputs_paths, maskcode=self.output_code,
        #                            crop=self.crop, cropshift=self.cropshift, testmode=True)
        # self.dataloader = DataLoader(self.dataset, batch_size=CFG.test_batch_size, shuffle=False)

        df = pd.DataFrame(glob(f"{CFG.project_root}/{self.datasets_folder}/images/*.npy"), columns=['image_path'])
        df['mask_path'] = df.image_path.str.replace('images', 'masks')
        df['id'] = df.image_path.map(lambda x: x.split('/')[-1].replace('.npy', ''))
        df['case'] = df.id.map(lambda x: x.split('_')[1])
        df['day'] = df.id.map(lambda x: x.split('_')[3])

        # remove faulty cases  # ZONA - this should be in the preprocessing step !!!
        fault1 = 'case_7_day_0'  # zona
        fault2 = 'case_81_day_30'  # zona
        df = df[~df['id'].str.contains(fault1) & ~df['id'].str.contains(fault2)].reset_index(drop=True)
        print(df.info())

        self.dataset = DatasetUWMadison3D(df, transforms=data_transforms['test'])

        self.dataloader = DataLoader(self.dataset, batch_size=CFG.test_batch_size,
                                     num_workers=1, shuffle=False, pin_memory=True)

        # Losses dataframe:
        self.losses = pd.DataFrame(columns=['subject', 'scan', 'loss type', 'loss'])

        # Run testing:
        self.test()

        # Save testing stats:
        # self.save_stats()

        # Backup results to S3 bucket:
        # if self.s3backup:
        #     self.backup_to_s3()

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def test(self):
        print(f'''
        ###########################################################################
                            >>>   Starting testing   <<<
        Model to be tested:                     {self.saved_model_path}
        Number of examples:                     {len(self.dataset)}
        Number of batches:                      {len(self.dataloader)}
        Batch size:                             {CFG.test_batch_size}

        S3 folder:                              {CFG.s3_results_folder}
        ###########################################################################
        ''')
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()

        folder_name = f"{CFG.project_root}/{CFG.ec2_results_folder}"
        os.makedirs(folder_name, exist_ok=True)

        for i, data_batch in enumerate(self.dataloader):

            inputs_cpu, targets_cpu, ids, cases, days = data_batch

            inputs_cpu = torch.unsqueeze(inputs_cpu, 0)
            inputs_cpu = torch.permute(inputs_cpu, (1, 0, 2, 3, 4))
            inputs = inputs_cpu.to(self.device, dtype=torch.float32)

            targets_cpu = torch.unsqueeze(targets_cpu, 0)
            targets_cpu = torch.permute(targets_cpu, (1, 0, 2, 3, 4))
            targets = targets_cpu.to(self.device, dtype=torch.float32)

            with torch.no_grad():
                outputs = self.model(inputs)
                batch_losses = self.criterion(outputs, targets)

                outputs = outputs.cpu().detach().numpy()

                for output, id in zip(outputs, ids):
                    np.save(f"{folder_name}/{id}.npy", output[0,:])
        
        print(f"wrote mask predictions to {folder_name}")

        # # .....................................................................................................

        # for i in range(len(paths)):
        #     output = outputs[i, 0, ...].cpu().numpy()
        #     target = targets[i, 0, ...].cpu().numpy().astype('uint8')
        #     loss = batch_losses[i].cpu().numpy()
        #     shape = shapes[i, ...].numpy()
        #     cc = crops_coords[i, ...].numpy()  # cc: crop coordinates
        #     affine = affines[i, ...].numpy()
        #     path = paths[i]
        #     # .................................................................................................
        #     output_nc = np.zeros(shape)  # nc: non-cropped
        #     output_nc[cc[0, 0]: cc[0, 1], cc[1, 0]: cc[1, 1], cc[2, 0]: cc[2, 1]] = output

        #     target[0, :, :] = target[-1, :, :] = \
        #         target[:, 0, :] = target[:, -1, :] = \
        #         target[:, :, 0] = target[:, :, -1] = 1  # mark edges of the crop box

        #     target_nc = np.zeros(shape)
        #     target_nc[cc[0, 0]: cc[0, 1], cc[1, 0]: cc[1, 1], cc[2, 0]: cc[2, 1]] = target
        #     # .................................................................................................
        #     '''
        #     Example of a path:
        #     /home/arman_avesta/capsnet/data/images/033_S_0725/2008-08-06_13_54_42.0/aparc+aseg_brainbox.mgz
        #     '''
        #     path_components = path.split('/')
        #     subject, scan = path_components[-3], path_components[-2]
        #     folder = join(CFG.project_root, self.niftis_folder, subject, scan)
        #     os.makedirs(folder, exist_ok=True)

        #     save_nifti(join(folder, 'output.nii.gz'), output_nc, affine)
        #     save_nifti(join(folder, 'target.nii.gz'), target_nc, affine)
        #     # .................................................................................................
        #     scan_loss = pd.DataFrame({'subject': subject, 'scan': scan,
        #                               'loss type': self.criterion, 'loss': [loss]})
        #     scan_loss.to_csv(join(folder, 'loss.csv'), index=False)

        #     self.losses = pd.concat([self.losses, scan_loss])

        # # .....................................................................................................

        # data_batches.set_description(f'Testing (loss: {self.losses["loss"].mean(): .3f}')

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def save_stats(self):
        """
        This function writes the testing results onto csv files.
        Outputs:
            - testing_losses.csv
            - testing_times.csv (computation times)
            - testing_hyperparameters.csv
            These files will be saved in the path set by self.results_folder
        """
        os.makedirs(join(CFG.project_root, self.niftis_folder), exist_ok=True)
        computation_time = datetime.now() - self.start_time

        hyperparameters = pd.DataFrame(index=['date and time',
                                              'segmentation target',
                                              'freesurfer code for segmentation target',
                                              'image crop size',
                                              'crop shift in (L,A,S) system',
                                              '-----------------------------------------------',
                                              'testing on:',
                                              'number of examples',
                                              'batch size',
                                              '-----------------------------------------------',
                                              'total computation time',
                                              'computation time per example',
                                              '-----------------------------------------------',
                                              'loss function',
                                              'loss',
                                              '-----------------------------------------------',
                                              'model',
                                              '-----------------------------------------------',
                                              'S3 NIfTIs folder',
                                              'inputs',
                                              'outputs'],
                                       data=[datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
                                             self.output_structure,
                                             self.output_code,
                                             self.crop,
                                             self.cropshift,
                                             '-----------------------------------------------',
                                             self.set,
                                             len(self.dataset),
                                             CFG.test_batch_size,
                                             '-----------------------------------------------',
                                             computation_time,
                                             computation_time / len(self.dataset),
                                             '-----------------------------------------------',
                                             self.criterion,
                                             self.losses['loss'].mean(),
                                             '-----------------------------------------------',
                                             self.model,
                                             '-----------------------------------------------',
                                             CFG.s3_results_folder,
                                             self.inputs_paths,
                                             self.outputs_paths])

        hyperparameters.to_csv(join(CFG.project_root, self.niftis_folder,
                                    self.hyperparameters_file), header=False)
        self.losses.to_csv(join(CFG.project_root, self.niftis_folder,
                                'scans_losses.csv'), index=False)

        print(f">>>   Testing loss: {self.losses['loss'].mean(): .3f}   <<<")

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def load_model(self):
        checkpoint = torch.load(self.saved_model_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        print(f'>>>   Loaded the model from: {self.saved_model_path}   <<<')

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def backup_to_s3(self, verbose=False):
        """
        This method backs up the results to S3 bucket.
        """
        ec2_folder = join(CFG.project_root, self.niftis_folder)
        command = f'aws s3 sync {ec2_folder} {CFG.s3_results_folder}' if verbose \
            else f'aws s3 sync {ec2_folder} {CFG.s3_results_folder} >/dev/null &'

        os.system(command)
        print('>>>   S3 backup done   <<<')


# ------------------------------------------ Run TrainUNet3D Instance ------------------------------------------

# Test the network:
utest = TestCapsNet3D()
