function [ out_hist ] = calculateDescriptor( img,par,w2c,codebook,mode )
% Given an input image, this code calculates a 8000-dim feature vector
% Currently, our code only supports extracting CN feature
maxK_knn = par.maxK_knn;
step = par.step;
cellW = par.cellW;
cellH = par.cellH;
imgH = size(img,1); % height of image
imgW = size(img,2); %width of image
cellX = (imgW-cellW)/step+1;
cellY = (imgH-cellH)/step+1;
nwords = size(codebook, 1);
    
Words = zeros(maxK_knn,cellY,cellX); %words name
Dwords = zeros(maxK_knn,cellY,cellX); %distance of words

%% for each patch, extract CN descriptor, followed by quantization with MA = 10
if(strcmp(mode,'CN'))
    idf = par.idf; 
    ne = par.ne;
    fLength = 11;
    Feature = zeros(fLength,cellY,cellX);
    img = double(img);
    for j = 1:cellY
        for i = 1:cellX
            data = img(((j-1)*step+1):((j-1)*step+cellH),((i-1)*step+1):((i-1)*step+cellW),:);% extract one cell
            tempCN = im2c(data, w2c, -2); %
            tempbin = reshape(mean(mean(tempCN)),1,[]); % feature bin
            tempnorm = zeros(nwords,1); % distance between nth words and the testing image
            for k = 1:nwords
                tempnorm(k) = norm(tempbin-codebook(k,:));
            end
            [D, order] = sort(tempnorm);
            Feature(:,j,i) = tempbin;
            Words(:,j,i) = order(1:maxK_knn);
            Dwords(:,j,i) = D(1:maxK_knn);
        end
    end
end

%% Gaussian mask for background suppression
flag_bu = par.flag_bu;
flag_gauss = par.flag_gauss;
ystep = par.ystep;% ystep means how many cells in y direction within one stripe
striplength = par.striplength;% same here
nstrip = floor((cellY-striplength)/ystep)+1;
k_knn = par.k_knn;
sigma = par.sigma;
Wwords = exp(-Dwords/(sigma^2));
if(flag_gauss == true)
    w_g_m = zeros(size(Wwords));
    [x,y] = meshgrid(linspace(-1,1,cellX),linspace(-1,1,cellY));
    sigmax = par.sigmax;
    sigmay = par.sigmay;
    w_g = exp(-(x/sigmax).^2-(y/sigmay).^2);
    for i = 1:cellX
        for j = 1:cellY
            w_g_m(:,j,i,:) = w_g(j,i);
        end
    end
    Wwords = Wwords.*w_g_m;
end

%% calculate final feature vector for an image
if k_knn<10
    Wwords(k_knn+1:end,:,:) = 0;
end
temphist = zeros(nwords,nstrip);
histword = zeros(cellY,nwords);
index = 0;
for j = 1:cellY
    for k = 1:cellX
        index = index+1;
        histword(j,Words(1:k_knn,j,k)) = (histword(j,Words(1:k_knn,j,k))+reshape(Wwords(1:k_knn,j,k),1,[]));
    end
end
for j = 1:(nstrip-1)
    temphist(:, j) = sum(histword(((j-1)*ystep+1):((j-1)*ystep+striplength), :));
end
temphist(:, nstrip) = sum(histword(((nstrip-1)*ystep+1):end, :));

TF = reshape((ones(nstrip, 1)*hist(reshape(Words(1:k_knn, :, :),1,[]),1:nwords))',[],1);
Hist = reshape(temphist, 1, [])';
findindex = find(TF == 0);
TF(findindex) = TF(findindex)+1;
if(flag_bu == true)
    Hist = Hist./(TF.^0.5);
end
out_hist = (Hist-ne).*sqrt(idf);


