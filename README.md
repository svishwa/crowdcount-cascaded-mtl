# CNN-based Cascaded Multi-task Learning of High-level Prior and Density Estimation for Crowd Counting  (Single Image Crowd Counting)

This is implementation of the paper [CNN-based Cascaded Multi-task Learning of High-level Prior and Density Estimation for Crowd Counting](https://arxiv.org/pdf/1707.09605.pdf) for single image crowd counting which is accepted at [AVSS 2017](http://www.avss2017.org/)

# Installation
1. Install pytorch
   
2. Clone this repository
  ```Shell
  git clone https://github.com/svishwa/crowdcount-cascaded-mtl.git
  ```
  We'll call the directory that you cloned crowdcount-cascaded-mtl `ROOT`


# Data Setup
1. Download ShanghaiTech Dataset

   Dropbox: https://www.dropbox.com/s/fipgjqxl7uj8hd5/ShanghaiTech.zip?dl=0 
   
   Baidu Disk: http://pan.baidu.com/s/1nuAYslz
   
2. Create Directory 
  ```Shell
  mkdir ROOT/data/original/shanghaitech/  
  ```
3. Save "part_A_final" under ROOT/data/original/shanghaitech/
4. Save "part_B_final" under ROOT/data/original/shanghaitech/
5. cd ROOT/data_preparation/

   run create_gt_test_set_shtech.m in matlab to create ground truth files for test data
6. cd ROOT/data_preparation/

   run create_training_set_shtech.m in matlab to create training and validataion set along with ground truth files

# Test
1. Follow steps 1,2,3,4 and 5 from Data Setup
2. Download pre-trained model files:

   [[Shanghai Tech A](https://www.dropbox.com/s/irho4laltre9ir5/cmtl_shtechA_204.h5?dl=0)]
   
   [[Shanghai Tech B](https://www.dropbox.com/s/lkt5ipshibs027w/cmtl_shtechB_768.h5?dl=0)]
   
   Save the model files under ROOT/final_models
   
3. Run test.py

	a. Set save_output = True to save output density maps
	
	b. Errors are saved in  output directory

# Training
1. Follow steps 1,2,3,4 and 6 from Data Setup
2. Run train.py


# Training with TensorBoard
With the aid of [Crayon](https://github.com/torrvision/crayon),
we can access the visualisation power of TensorBoard for any 
deep learning framework.

To use the TensorBoard, install Crayon (https://github.com/torrvision/crayon)
and set `use_tensorboard = True` in `ROOT/train.py`.

# Other notes
1. During training, the best model is chosen using error on the validation set. 
2. 10% of the training set is set aside for validation. The validation set is chosen randomly.
3. Following are the results on  Shanghai Tech A and B dataset:
			
                |     |  MAE  |   MSE  |
                ------------------------
                | A   |  101  |   148  |
                ------------------------
                | B   |   17  |    29  |
                
   It may be noted that the results are slightly different from the paper. This is due to a few implementation differences as the earlier implementation was in torch-lua. Contact me if torch models (that were used for the paper) are required.


           

