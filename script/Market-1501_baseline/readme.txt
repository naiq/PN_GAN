%********************************************************************************************************************************************
% This code provides performance evaluation on Market-1501 dataset. We use the BoW method described in the                                                        %       
% following paper,                                                                                                 
%                                                                                                                  
% Liang Zheng*, Liyue Shen*, Lu Tian*, Shengjin Wang, Jingdong Wang, and Qi Tian. Scalable Person Re-identification: 
% A Benchmark, ICCV 2015.(*equal contribution)                                               
%                                                                                                                     
% If you find this code useful, please kindly cite our paper.                                                          
%*******************************************************************************************************************************************

The current version fixes bugs in the previous version. In this version, "gt_query" folder is not used: ground truths are generated automatically.

1. To run this code, please first download the Market-1501 dataset, and unzip it in the "dataset" folder.

2. Run "baseline_evaluation.m". It extracts features for both queries and database images, and uses inner product (Euclidean distance) as similarity measurement.
The function produces mAP and CMC curve on the dataset.

Compared with previous version, we add the following components:

1) multiple queries by max and avg pooling

2) search re-ranking

3) re-id between camera pairs (a confusion matrix is calculated) 

3. Run "metric_learning_evaluation.m". It generate positive and negative training data, and various metric learning methods are applied. mAP and CMC curve are reported.

Note: for metric learning, we largely follow the code provided by [R1].
[R1] M. Koestinger et al, Large scale metric learning from equivalence constraints, CVPR 2012.


If you have any problem, please contact me at liangzheng06@gmail.com.