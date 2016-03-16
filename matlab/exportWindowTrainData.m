% Traverses WDD-Folder-Structure  and creates for all dances with Ground
% Truth Data a Trainings- and a Validationmatrix.
% Since it traverses all dances, the matrices increase with every dance, 
% which for a high number of dances can cause performance issues in 
% performance. Therefore there is the possibilty to temporarly split the 
% matrices up into splitNumber parts, which can improve the performance for 
% a large number of dances. At the end they get merged together.
% Arguments:
%   folderPath: folder where the dances can be found
%   outputFile: name of the .mat file in which the matrix will be saved
%   pixel: the height and width of the images in the folder
%   minWindow: how many images of a dance should be in one sample
%   splitNumber: Number of how many times to split the matrix
%   splitSize: Number of how many dances are max in one splitted up part
%
% splitSize*splitNumber should be roughly the number of dances. 
% The outputfile has following contents:
%   X_train: Matrix with 30-Frame samples that should be used for training
%   Y_train: Labels for samples in X_train
%   X_test: Matrix with 30-Frame samples that should be used for validation
%   Y_test: Labels for samples in X_test
function exportWindowTrainData(folderPath, outputFile, pixel, minWindow, splitNumber, splitSize)
folder = dir(folderPath);
UD.currentImgFolder = 1;
UD.imgFolders = {};
progress = 0;
X_train = [];
Y_train = [];
X_test = [];
Y_test = [];
n = splitNumber;
part_size = splitSize;
nActual = 0;
t = 1;
s = 1;
% Traverse folder structure
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
                % See in the Ground Truth data if image sequence belongs to a dance or not
                groundTruthFileName = strcat(imgfolderpath,'/gt.csv');
                groundTruthFile = fopen(groundTruthFileName);
                if (groundTruthFile == -1)
                    % If there is no GroundTruth for this Data we give it
                    % an Error value and do not use it.
                    L = -1; 
                else
                    TS = textscan(groundTruthFile, '%s');
                    L = -1;
                    if strcmp(TS{1,1},'j')
                        L = 1;
                    else if strcmp(TS{1,1},'n')
                            L = 0;
                        end
                    end
                    fclose(groundTruthFile);
                end
                % Read all images in this dance
                WimageMatrix = uint8(zeros(numel(length(UD.imageArray)),pixel,pixel));
                for k = 1:numberOfImages
                    P = imread(fullfile(imgfolderpath,UD.imageArray(k).name));
                    W = P(:,:,1);
                    WimageMatrix(k,:,:) = W;
                end
                imageMatrices{1} = WimageMatrix;
                nb_imageMatrices = 1;
                % Make Sliding-Window-Matrix from all the images in this dance.
                p = rand;
                zeros4D = uint8(zeros(nb_imageMatrices*(numberOfImages-minWindow),minWindow,pixel,pixel));
                zeros1D = uint8(zeros(1,nb_imageMatrices*(numberOfImages-minWindow)));
                % Decide if dance will be part of Train- or Testmatrix
                if (p<0.8)
                    X_train(t:t+nb_imageMatrices*(numberOfImages-minWindow)-1,:,:,:) = zeros4D;
                    Y_train(t:t+nb_imageMatrices*(numberOfImages-minWindow)-1,1) = zeros1D;
                    for i = 1:nb_imageMatrices
                        imageMatrix = imageMatrices{i};
                        for k = 1:(numberOfImages-minWindow)
                            X_train(t,:,:,:) = imageMatrix(k:k+minWindow-1,:,:);
                            Y_train(t,:) = L;
                            t = t+1;
                        end
                    end
                else
                    X_test(s:s+nb_imageMatrices*(numberOfImages-minWindow)-1,:,:,:) = zeros4D;
                    Y_test(s:s+nb_imageMatrices*(numberOfImages-minWindow)-1,1) = zeros1D;
                    for i = 1:nb_imageMatrices
                        imageMatrix = imageMatrices{i};
                        for k = 1:(numberOfImages-minWindow)
                            X_test(s,:,:,:) = imageMatrix(k:k+minWindow-1,:,:);
                            Y_test(s,:) = L;
                            s = s+1;
                        end
                    end
                end  
                % Splitting up the matrix every splitSize Dances, but not
                % more than splitNumber times.
                for i = 1:n
                    if (progress == i*part_size)
                        nActual = nActual +1; 
                        irand = randperm(size(X_train,1));
                        featuresTrain = X_train(irand,:,:,:);
                        targetsTrain = Y_train(irand,:);
                        X_train = featuresTrain;
                        Y_train = targetsTrain;
                        clear featuresTrain targetsTrain
                        save(strcat('part', int2str(i) ,'.mat'), 'X_train', 'Y_train', 'X_test', 'Y_test', '-v7.3');
                        s = 1;
                        t = 1;
                        clear X_test Y_test X_train Y_train
                    end
                end         
            end
        end
    end
end
% Saving the last part of the splitted up matrix.
if (progress > n*part_size || progress < n*part_size)
    irand = randperm(size(X_train,1));
    featuresTrain = X_train(irand,:,:,:);
    targetsTrain = Y_train(irand,:);
    X_train = featuresTrain;
    Y_train = targetsTrain;
    clear featuresTrain targetsTrain
    save('last_part.mat', 'X_train', 'Y_train', 'X_test', 'Y_test', '-v7.3');
end

% Merging all the splitted up parts together
AX_train = [];
AY_train = [];
AX_test = [];
AY_test = [];
for i = 1:nActual
    load(strcat('part', int2str(i) ,'.mat'));
    AX_train = [AX_train; X_train];
    AY_train = [AY_train; Y_train];
    AX_test = [AX_test; X_test];
    AY_test = [AY_test; Y_test];
    clear X_test Y_test X_train Y_train
end

load('last_part.mat')

AX_train = [AX_train; X_train];
AY_train = [AY_train; Y_train];
AX_test = [AX_test; X_test];
AY_test = [AY_test; Y_test];
clear X_test Y_test X_train Y_train

% Save final matrix
irand = randperm(size(AX_test,1));
X_test = AX_test(irand,:,:,:);
Y_test = AY_test(irand,:);
clear AX_test AY_test irand
irand = randperm(size(AX_train,1));
X_train = AX_train(irand,:,:,:);
Y_train = AY_train(irand,:);
clear AX_train AY_train irand
save(outputFile, 'X_test', 'Y_test', 'X_train', 'Y_train', '-v7.3')

% Deleting the splitted up parts again clearing the workspace
for i = 1:nActual
    delete(strcat('part', int2str(i) ,'.mat'))
end
delete('last_part.mat')
clear all
