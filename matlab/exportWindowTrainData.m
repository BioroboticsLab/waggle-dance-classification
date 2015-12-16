clearvars
folderPath='/Users/mehmedhalilovic/Documents/MATLAB/BA/GroundTruth13122015S';
folder = dir(folderPath);
UD.currentImgFolder = 1;
UD.imgFolders = {};
minWindow = 30; %sliding window size
pixel = 30; %width and height of imgages, if n x n image --> pixel = n
progress = 0; %number of dances that have been processed 
t = 1;
s = 1;
%traverse folder structure
for d = folder'
    if d.isdir && ~strcmp(d.name,'.') && ~strcmp(d.name,'..')
        subpath = fullfile(folderPath,d.name);
        subfolder = dir(subpath);
        for f = subfolder'
            if f.isdir && ~strcmp(f.name,'.') && ~strcmp(f.name,'..')
                imgfolderpath = fullfile(subpath, f.name);
                UD.imgFolders{end+1} = imgfolderpath;
                UD.imageArray   = dir(fullfile(imgfolderpath,'*.png'));
                numberOfImages = length(UD.imageArray);
                progress = progress + 1
                %Check in GroundTruth data whether image sequence belongs to a dance or not
                groundTruthFileName = strcat(imgfolderpath,'/gt.csv');
                groundTruthFile = fopen(groundTruthFileName);
                %If there is no GroundTruth for this data it gets an error value
                if (groundTruthFile == -1)
                    L = -1; 
                else
                    TS = textscan(groundTruthFile, '%s'); %invalid
                    L = -1;
                    if strcmp(TS{1,1},'j') %positive
                        L = 1;
                    else if strcmp(TS{1,1},'n') %negative
                            L = 0;
                        end
                    end
                    fclose(groundTruthFile);
                end
                % read all images in this dance
                imageMatrix = uint8(zeros(numel(length(UD.imageArray)),pixel,pixel));
                for k = 1:numberOfImages
                     W = imread(fullfile(imgfolderpath,UD.imageArray(k).name));
                     P = W(:,:,1);
                     imageMatrix(k,:,:) = P;
                end
                
                % Add Sliding-Window-Matrixes from all the images to the data.
                p = rand;
                zeros4D = uint8(zeros(numberOfImages-minWindow,minWindow,pixel,pixel));
                zeros1D = uint8(zeros(1,numberOfImages-minWindow));
                % decide if dance will be part of train or test
                if (p<=0.8)
                    X_train(t:t+(numberOfImages-minWindow-1),:,:,:) = zeros4D;
                    Y_train(t:t+(numberOfImages-minWindow-1),1) = zeros1D;
                    for k = 1:(numberOfImages-minWindow)
                        E = imageMatrix(k:k+minWindow-1,:,:);     
                        X_train(t,:,:,:) = E;
                        Y_train(t,:) = L;
                        t = t+1;
                    end
                else
                    X_test(s:s+(numberOfImages-minWindow-1),:,:,:) = zeros4D;
                    Y_test(s:s+(numberOfImages-minWindow-1),1) = zeros1D;
                    for k = 1:(numberOfImages-minWindow)
                        E = imageMatrix(k:k+minWindow-1,:,:);
                        X_test(s,:,:,:) = E;
                        Y_test(s,:) = L;
                        s = s+1;
                    end
                end
            end
        end
    end
end

% irand = randperm(size(X_train,1));
% featuresTrain = X_train(irand,:,:,:);
% targetsTrain = Y_train(irand,:);
% irand = randperm(size(X_test,1));
% featuresTest = X_test(irand,:,:,:);
% targetsTest = Y_test(irand,:);
% borderTrainTest = size(X_train,1);
% features = [featuresTrain; featuresTest];
% targets = [targetsTrain; targetsTest];


save('data2015S.mat', 'X_train', 'Y_train', 'X_test', 'Y_test', '-v7.3');
