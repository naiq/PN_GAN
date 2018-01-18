# PN_GAN
This 'test_file' contains four folders:
    
	- dataset: it contains all synthesized images of VIPeR dataset; due to the size of files, I upload the jpeg files via  
	https://drive.google.com/open?id=1edWARIlBcaRFNrrgehUZhnoxXbsmQsKI
    - feature: where to save the features of probe and gallery
    - model: it has two files, caffemodel and deploy.prototxt
    - script: it contains two python files, one text file and one matlab folder. Python files are used for calculating cmc score with python code (I modify it based on tong xiao's code.), text file is the data list for testing, matlab folder is used for calculating cmc score with matlab code (I modify it based on zhedong's code.)

How to run it:
(1) run 'viper_feature.py' to extract features of probe and gallery, the features will be saved in folder '../feature/';
(2) run 'CMC_viper.py' to compute cmc scores with python code, it will output three kinds of results: 
    - avg: 8 pose features are fused by average operation
    - max: 8 pose features are fused by maximum operation
    - concat: 8 pose features are fused by concatenation operation
(3) (optional) run 'Market-1501_baseline/zzd_evaluation_res_faster.m' to compute cmc scores with matlab code. You can modify the code in line 93 to obtain different result of each metric learning (e.g. 'dist_avg.mat', 'dist_max.mat', or 'dist_concat.mat'). It should get the same results with step 2.

Note: (1) The code in line 121 and 122 in 'viper_feature.py' is the path to viper dataset, it should be modified
      (2) The code in line 69 in 'CMC_viper.py' is the path to viper dataset, it should be modified, too
	  (3) The model we provide is trained for 10000 iteration on Market-1501 dataset with the purpose of showing the effectiveness of our model; our full model (the results reported in the paper) is trained with much more itration.
	  
	  
	  
	  

#Acknowledgment:

The testing codes are modified from Tong Xiao's code. 

 

