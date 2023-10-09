The code here performs 3D modeling of GI-Tract MRI scans from University of Wisconson Madison (UWM) image segmentation challenge on Kaggle.
The competition can be found at:
* https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/


The code here is a combination of two sources:
Source 1: an example provided by Kaggle competitor "AWSAF49" from the Kaggle competition.
* https://www.kaggle.com/code/awsaf49/uwmgi-mask-data

Source 2: 
The repository was originally forked from
https://github.com/Aneja-Lab-Yale/Aneja-Lab-Public-3D2D-Segmentation
Reference Publication
https://www.researchgate.net/publication/367985871_Comparing_3D_25D_and_2D_Approaches_to_Brain_Image_Auto-Segmentation
<also see supplemental materials>
Both files are copied in this folder...

Changes Summary:
To apply the Aneja-Lab-Yale code for the UWM GI-Tract dataset, the following were the most significant changes:
a) a preprocessing script for UWM dataset is added - this is found in /pre_processing_uwmadison
b) a new Dataset class (DatasetUWMadison) was introduced in data_set_uwmadiswon.py to accomodate the new data
c) changes were made to unet_train_2d.py to use the new Dataset

To Run:
Download the UWMadison dataset
Follow steps in /pre_processing_uwmadison to preprocess
Train using unet_train.2d.py


+===== status as of 6-October-2023 =================
trying to use UWMadison data...

+===== status as of 24-Sept-2023 =================
forked the repo and the code looks really really nice.
Unable, however to make sense of the LONDI dataset...  trying to run unet_train_2d and it's looking for data/datasets/train_inputs_120.csv'
But who really cares - I'm not interested in the LONDI dataset, rather I want to target the UWMadison GI Tract data....
so recommendation is to pursue that direction....


