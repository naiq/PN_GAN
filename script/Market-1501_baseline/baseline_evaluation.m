clc;clear all;close all;
%***********************************************%
% This code runs on the Market-1501 dataset.    %
% Please modify the path to your own folder.    %
% We use the mAP and hit-1 rate as evaluation   %
%***********************************************%
% if you find this code useful in your research, please kindly cite our
% paper as,
% Liang Zheng, Liyue Sheng, Lu Tian, Shengjin Wang, Jingdong Wang, and Qi Tian,
% Scalable Person Re-identification: A Benchmark, ICCV, 2015.

% Please download Market-1501 dataset and unzip it in the "dataset" folder.

%% load data for feature extraction
codebooksize = 350;
codebook = importdata(['data\codebook_' num2str(codebooksize) '.mat']);
par = importdata(['data\params_' num2str(codebooksize) '.mat']);
w2c = importdata('data\w2c.mat'); % used in CN extraction

%% add necessary paths
query_dir = 'dataset\query\';% query directory
test_dir = 'dataset\bounding_box_test\';% database directory
gt_dir = 'dataset\gt_bbox\'; % directory of hand-drawn bounding boxes
addpath 'CM_curve\' % draw confusion matrix

%% calculate query features
query_files = dir([query_dir '*.jpg']);
if ~exist('Hist_query.mat');
    Hist_query = CalFeatFromDir(query_dir, par, w2c, codebook);
    save('Hist_query.mat', 'Hist_query');
else
    Hist_query = importdata('Hist_query.mat');
end
nQuery = size(Hist_query, 2);

%% calculate database features
test_files = dir([test_dir '*.jpg']);
if ~exist('Hist_test.mat');
    Hist_test = CalFeatFromDir(test_dir, par, w2c, codebook);
    save('Hist_test.mat', 'Hist_test');
else
    Hist_test = importdata('Hist_test.mat');
end
nTest = size(Hist_test, 2);

%% calculate the ID and camera for database images
test_files = dir([test_dir '*.jpg']);
testID = zeros(length(test_files), 1);
testCAM = zeros(length(test_files), 1);
if ~exist('data\testID.mat')
    for n = 1:length(test_files)
        img_name = test_files(n).name;
        if strcmp(img_name(1), '-') % junk images
            testID(n) = -1;
            testCAM(n) = str2num(img_name(5));
        else
            testID(n) = str2num(img_name(1:4));
            testCAM(n) = str2num(img_name(7));
        end
    end
    save('data\testID.mat', 'testID');
    save('data\testCAM.mat', 'testCAM');
else
    testID = importdata('data\testID.mat');
    testCAM = importdata('data\testCAM.mat');    
end

%% calculate the ID and camera for query images
query_files = dir([query_dir '*.jpg']);
queryID = zeros(length(query_files), 1);
queryCAM = zeros(length(query_files), 1);
if ~exist('data\queryID.mat')
    for n = 1:length(query_files)
        img_name = query_files(n).name;
        if strcmp(img_name(1), '-') % junk images
            queryID(n) = -1;
            queryCAM(n) = str2num(img_name(5));
        else
            queryID(n) = str2num(img_name(1:4));
            queryCAM(n) = str2num(img_name(7));
        end
    end
    save('data\queryID.mat', 'queryID');
    save('data\queryCAM.mat', 'queryCAM');
else
    queryID = importdata('data\queryID.mat');
    queryCAM = importdata('data\queryCAM.mat');    
end

%% calculate features for multiple queries
if ~exist('Hist_query_max.mat');
    Hist_max = []; % multiple queries by max pooling
    Hist_avg = []; % multiple queries by avg pooling
    for n = 1:length(query_files)
        n
        img_name = query_files(n).name;
        gt_files = dir([gt_dir img_name(1:7) '*.jpg']);
        tmp_feature = [];
        for m = 1:length(gt_files)
            img_path = [gt_dir gt_files(m).name];
            img = imread(img_path);
            tmp_feature(:, m) = calculateDescriptor(img,par,w2c,codebook,'CN');
            tmp_feature(:, m) = tmp_feature(:, m)./sqrt(sum(tmp_feature(:, m).^2));
        end
        Hist_max(:, n) = max(tmp_feature, [], 2);
        Hist_avg(:, n) = mean(tmp_feature, 2);
    end
    save('Hist_query_max.mat', 'Hist_max');
    save('Hist_query_avg.mat', 'Hist_avg');
else
    Hist_max = importdata('Hist_query_max.mat');
    Hist_avg = importdata('Hist_query_avg.mat');
end
% another normalization
sum_val = sqrt(sum(Hist_max.^2));
sum_val = repmat(sum_val, [size(Hist_max,1), 1]);
sum_val2 = sqrt(sum(Hist_avg.^2));
sum_val2 = repmat(sum_val2, [size(Hist_avg,1), 1]);
Hist_max = Hist_max./sum_val;
Hist_avg = Hist_avg./sum_val2;

%% search the database and calcuate re-id accuracy
ap = zeros(nQuery, 1); % average precision
ap_max = zeros(nQuery, 1); % average precision with MultiQ_max 
ap_avg = zeros(nQuery, 1); % average precision with MultiQ_avg 
ap_max_rerank  = zeros(nQuery, 1); % average precision with MultiQ_max + re-ranking 
ap_pairwise = zeros(nQuery, 6); % pairwise average precision with single query (see Fig. 7 in the paper)

CMC = zeros(nQuery, nTest);
CMC_max = zeros(nQuery, nTest);
CMC_avg = zeros(nQuery, nTest);
CMC_max_rerank = zeros(nQuery, nTest);

r1 = 0; % rank 1 precision with single query
r1_max = 0; % rank 1 precision with MultiQ_max
r1_avg = 0; % rank 1 precision with MultiQ_avg
r1_max_rerank = 0; % rank 1 precision with MultiQ_max + re-ranking
r1_pairwise = zeros(nQuery, 6);% pairwise rank 1 precision with single query (see Fig. 7 in the paper)

dist = sqdist(Hist_test, Hist_query); % distance calculate with single query. Note that Euclidean distance is equivalent to cosine distance if vectors are l2-normalized
dist_max = sqdist(Hist_test, Hist_max); % distance calculation with MultiQ_max
dist_avg = sqdist(Hist_test, Hist_avg); % distance calculation with MultiQ_avg
dist_cos_max = (2-dist_max)./2; % cosine distance with MultiQ_max, used for re-ranking

knn = 1; % number of expanded queries. knn = 1 yields best result
queryCam = importdata('data\queryCam.mat'); % camera ID for each query
testCam = importdata('data\testCam.mat'); % camera ID for each database image

for k = 1:nQuery
    k
    % load groud truth for each query (good and junk)
    good_index = intersect(find(testID == queryID(k)), find(testCAM ~= queryCAM(k)))';% images with the same ID but different camera from the query
    junk_index1 = find(testID == -1);% images neither good nor bad in terms of bbox quality
    junk_index2 = intersect(find(testID == queryID(k)), find(testCAM == queryCAM(k))); % images with the same ID and the same camera as the query
    junk_index = [junk_index1; junk_index2]';
    tic
    score = dist(:, k);
    score_avg = dist_avg(:, k); 
    score_max = dist_max(:, k);
    
    % sort database images according Euclidean distance
    [~, index] = sort(score, 'ascend');  % single query
    [~, index_max] = sort(score_max, 'ascend'); % multiple queries by max pooling
    [~, index_avg] = sort(score_avg, 'ascend'); % multiple queries by avg pooling
    
    [ap(k), CMC(k, :)] = compute_AP(good_index, junk_index, index);% compute AP for single query
    [ap_max(k), CMC_max(k, :)] = compute_AP(good_index, junk_index, index_max);% compute AP for MultiQ_max
    [ap_avg(k), CMC_avg(k, :)] = compute_AP(good_index, junk_index, index_avg);% compute AP for MultiQ_avg
    ap_pairwise(k, :) = compute_AP_multiCam(good_index, junk_index, index, queryCam(k), testCam); % compute pairwise AP for single query
    
    %%%%%%% re-ranking after "multiple queries with max pooling" %%%%%%%%%
    count = 0;
    score_cos_max = dist_cos_max(:, k);
    for i = 1:length(index_max) % index_max is the same with sorting images according to score_cos_max
        if ~isempty(find(junk_index == index_max(i), 1)) % a junk image
            continue;
        else
            count = count + 1;
            query_index = index_max(i);
            query_hist_new = single(Hist_test(:, query_index));% expanded query
            query_hist_new = repmat(query_hist_new, [1, size(Hist_test, 2)]);
            score_new = sum(query_hist_new.*Hist_test);
            score_cos_max = score_cos_max + score_new'./(count+1); % update score
            if count == knn % will break if "knn" queries are expanded
                break;
            end
        end
    end
    [~, index_max_rerank] = sort(score_cos_max, 'descend');
    [ap_max_rerank(k), CMC_max_rerank(k, :)] = compute_AP(good_index, junk_index, index_max_rerank); % compute AP for MultiQ_max + re-rank
    %%%%%%%%%%%%%%%%%%%%%%%% re-ranking %%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%% calculate pairwise r1 precision %%%%%%%%%%%%%%%%%%%%
    r1_pairwise(k, :) = compute_r1_multiCam(good_index, junk_index, index, queryCam(k), testCam); % pairwise rank 1 precision with single query
    %%%%%%%%%%%%%% calculate r1 precision %%%%%%%%%%%%%%%%%%%%
end
CMC_max_rerank = mean(CMC_max_rerank);
CMC_max = mean(CMC_max);
CMC_avg = mean(CMC_avg);
CMC = mean(CMC);
%% print result
fprintf('single query:                                   mAP = %f, r1 precision = %f\r\n', mean(ap), CMC(1));
fprintf('multiple queries with avg pooling:              mAP = %f, r1 precision = %f\r\n', mean(ap_avg), CMC_avg(1));
fprintf('multiple queries with max pooling:              mAP = %f, r1 precision = %f\r\n', mean(ap_max), CMC_max(1));
fprintf('multiple queries with max pooling + re-ranking: mAP = %f, r1 precision = %f\r\n', mean(ap_max_rerank), CMC_max_rerank(1));
[ap_CM, r1_CM] = draw_confusion_matrix(ap_pairwise, r1_pairwise, queryCam);
fprintf('average of confusion matrix with single query:  mAP = %f, r1 precision = %f\r\n', (sum(ap_CM(:))-sum(diag(ap_CM)))/30, (sum(r1_CM(:))-sum(diag(r1_CM)))/30);

%% plot CMC curves
figure;
s = 50;
CMC_curve = [CMC_max_rerank; CMC_max; CMC_avg; CMC ];
plot(1:s, CMC_curve(:, 1:s));




