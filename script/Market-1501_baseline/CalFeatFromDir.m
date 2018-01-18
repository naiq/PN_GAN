function features = CalFeatFromDir( img_dir, par, w2c, codebook )

img_files = dir([img_dir '*.jpg']);
nImg = length(img_files);
features = []; 
for n = 1:nImg % number of images
    n 
    img = imread([img_dir,img_files(n).name]);
    features(:,n) = calculateDescriptor(img, par, w2c, codebook, 'CN');
end

% normalization
Dim = size(features,1);
sum_val = sqrt(sum(features.^2));
sum_val = repmat(sum_val, [Dim, 1]);
features = features./sum_val;


end

