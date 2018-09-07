# PN_GAN
In current version, we release the codes of PN-GAN and re-id testing . The other parts of our project will be released later.

How to run it:

GAN:

(1) run 'GAN/train.py' to train the GAN model. The model and log file will be saved in folder 'GAN/model' and 'GAN/log' respectively. The validate images will be synthesized in 'GAN/images';

or (2) run 'GAN/evaluate.py' to generate images for specific testing image. The output will be saved in folder 'GAN/test'

Person re-id:
(1) run 'viper_feature.py' to extract features of probe and gallery, the features will be saved in folder '../feature/';

(2) run 'CMC_viper.py' to compute cmc scores with python code, it will output three kinds of results: 
    - avg: 8 pose features are fused by average operation
    - max: 8 pose features are fused by maximum operation
    - concat: 8 pose features are fused by concatenation operation

(3) (optional) run 'Market-1501_baseline/zzd_evaluation_res_faster.m' to compute cmc scores with matlab code. You can modify the code in line 93 to obtain different result of each metric learning (e.g. 'dist_avg.mat', 'dist_max.mat', or 'dist_concat.mat'). It should get the same results with step 2.

	  
	 	 
#Acknowledgment:

The testing codes are modified from Tong Xiao's code, and also refer to Zhedong Zheng's codes.

 

