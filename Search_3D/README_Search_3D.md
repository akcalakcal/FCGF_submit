
## Introduction

The provided code includes two main parts. The first part is training FCGF method and extraction of FCGF features of a point cloud.
The provided code is the modified version of the paper's code from "https://github.com/chrischoy/FCGF". The second part
is reading the FCGF features and computing the VLAD features together with performing query search. 

The main reason behind the partial implementation is them limitation of fo teh computation sources and I couldn't upgrade CUDA version 
of my office machine which I am connected remotely. I handle the Pytorch training and test parts using my universiy's computer cluster
which has no visualization capability. Then I handle the VLAD descriptor part and visualization of the query search in my office machine. 
If I had a chance to update my CUDA version, I would have designed this system as a complete and one framework. 


## Folder Structure

 * Main Folder
	* FCGF_submit
		* ...
		* scripts/
		* data/
			* threedmatch/
		* data_test/
			* tmp/
				* 7-scenes-redkitchen/
					* cloud_bin_0.ply
		* features_tmp/
			* 7-scenes-redkitchen/
				* cloud_bin_0.npz
		* outputs_trained_models/
		* README.md (Read it for the installation)
		* ...
		* Search_3D (This part is VLAD Descriptor and Query Search Part)
			* util/
			* VLADlib/
			* README_Search_3D.md
			* main_single_pcd.py
			* run_3D_search_single_pcd.sh

			


### 
#  Main Part - 1 (FCGF_submit)
#  Environment Setup - Training FCGF Network and FCGF Feature Extraction
###

## Requirements

- Ubuntu 16.04 or higher
- CUDA 10.0 or higher
- Python v3.7 or higher
- Pytorch v1.2 or higher
- [MinkowskiEngine](https://github.com/stanfordvl/MinkowskiEngine) v0.2.7 or higher


## Installation & Dataset Download

Setup anaconda environment (If it is necessary)

```
$ ./setup_new_virtual_env.sh
```



# Clone the project repository from "https://github.com/akcalakcal/FCGF_submit.git" under Main_Folder path (Please follow the folder structure above)
# This repostory has two main parts: "FCGF_submit" and "FCGF_submit/Search_3D".
$ git clone https://github.com/akcalakcal/FCGF_submit.git

# -> Follow the installation steps of required packages from "https://github.com/akcalakcal/FCGF_submit.git"
# Please read the README.md file under the repository : ${Main_Folder}/FCGF_submit/README.md
# -> After setting up the environment


# Download training dataset

$ ./scripts/download_datasets.sh ./data/

# Train FCGF - if you don't want to train FCGF please go to next step where you can download trained model
# Please set the paths in the script before running it. 
$ ./run_train_FCGF.sh

# Download trained model - if you dont want to train FCGF rom scrath, please use the pre-trained model 
# We train the model from scratch using 3DMatch dataset. In original FCGF repository, they don't provide pre-trained models.
# The provided models are trained for 4 days usign 3DMatch dataset.

# -> Please download trained models from this link "https://drive.google.com/drive/folders/1_JegPK_m86oDrUAJtxmAncauNwBUVTGz?usp=sharing"
# -> And locate the unzipped folder with a name "outputs_trained_models", as it is shown in "Folder Structure" above. 

# FCGF feature extraction part

# -> You can run the script below to extract features of a point cloud. 
# This script takes a point cloud input and extracts FCGF features of this  
# point cloud and save FCGF features in a folder. 
# Please set the folder paths correctly to run this script. 

$ ./run_single_pcd_FCGF_ext.sh

# If you can not run the feature extractor, don't worry. 
# There is a previusly computed FCGF features and point cloud pair available 
# under these directories. We can use them in the following steps. 
# 	point cloud -> ${Main_folder}/FCGF_submit/data_test/tmp/7-scenes-redkitchen/cloud_bin_0.ply
#	FCGF features -> ${Main_folder}/FCGF_submit/features_tmp/7-scenes-redkitchen/cloud_bin_0.npz


### 
#  Main Part - 2 (Search_3D)
#  VLAD Descriptor Extractor and Query Search
###

# Note. If you want to run only this part, you don't need to install PyTorch and Minkowski Engine. 


# If you fetch repository from "https://github.com/akcalakcal/FCGF_submit.git"
# you will have input point cloud and associated FCGF features under following paths
# 	point cloud -> ${Main_folder}/FCGF_submit/data_test/tmp/7-scenes-redkitchen/cloud_bin_0.ply
#	FCGF features -> ${Main_folder}/FCGF_submit/features_tmp/7-scenes-redkitchen/cloud_bin_0.npz

# Main Part - 2  can start given a point cloud and its FCGF features. 

# In order to perform the "VLAD descriptor extraction" and "Query search"
# Please run the script below
# Please change the folder paths and input files. 
# A video will be provided to show how to use the application. Please watch it before running the script. 

$ ./run_3D_search_single_pcd.sh

# After running this script, open3D visualization will open.
# You can move the point cloud using the mouse inputs.
# The black box is query box, and the blue boxes are the search results. 


# This is the end of README_Search_3D.md file. 

# For more details about the methodology and visuals, please refer to Report file. 












