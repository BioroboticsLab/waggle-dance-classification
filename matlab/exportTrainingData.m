function [] = exportTrainingData(source_folder, windowSize, train_proportion)
output_filename = 'train_test.mat'
folder_obj  = dir(source_folder);
nDirs       = length(folder_obj);
progress    = 0;
X = []; 
Y = [];
% Traverse folder structure
% first level corresponds to hour-minute-cam_id
fprintf('\n\n\n')
for d = folder_obj' 
progress = progress+1;
clearCharactersFromConsole(3);
fprintf('%2d%%', round(100*progress/nDirs));

    if d.isdir && ~strcmp(d.name,'.') && ~strcmp(d.name,'..')
        subpath         = fullfile(source_folder, d.name);
        subfolder_obj   = dir(subpath);
        % second level corresponds to sequence number of waggle run 
        for f = subfolder_obj'
            
            if f.isdir && ~strcmp(f.name,'.') && ~strcmp(f.name,'..')
                imgfolder_path = fullfile(subpath, f.name);
                
                % load label from ground truth file
                L = loadGroundTruth(imgfolder_path);
                
                % go to next folder if no ground truth could be found
                if L < 0 
                    continue;
                end
                
                ImageTensor = loadImages(imgfolder_path);
                
                % prepare data structures for current dance
                [dX] = createSlidingWindowTensor(ImageTensor, windowSize);
                dY = repmat(L, size(dX, 1), 1);
                
                % add this on the current batch
                X = [X; dX];
                Y = [Y; dY];             
            end
        end
    end
end

nData = size(X, 1);
idxSplit = floor(train_proportion * nData);

% shuffle data
perm = randperm(nData);
X = X(perm, :, :, :);
Y = Y(perm);

% split in train and test set
X_train = X(1:idxSplit, :, :, :);
Y_train = Y(1:idxSplit);
X_test  = X(idxSplit+1:end, :, :, :);
Y_test  = Y(idxSplit+1:end);

save(output_filename, 'X_train', 'Y_train', 'X_test', 'Y_test');


% this function takes a sequence of images and creates a 4D tensor
% of subsequences of length <window_size> images
function M = createSlidingWindowTensor(ImageTensor, windowSize)
[nFrames, imgDim, ~] = size(ImageTensor);
nWindows = (nFrames - windowSize) + 1;
M = uint8(zeros(nWindows, windowSize, imgDim, imgDim));
for k = 1 : nWindows  % ISSUE: should be +1
	M(k, :, :, :) = ImageTensor(k : k+windowSize-1, :, :);
end


function I = loadImages(folder)
filenames   = dir( [folder '/*.png'] );
nImages     = length(filenames);
dim         = size(imread([folder '/' filenames(1).name]), 1);
I           = uint8(zeros(nImages, dim, dim));
for k = 1:nImages
    P = imread(fullfile(folder,filenames(k).name));
    %take only one color channel (grayvalue src anyway)
    I(k,:,:) = P(:,:,1); 
end


function L = loadGroundTruth(folder)
gt_file = [folder '/gt.csv'];
if exist(gt_file, 'file')
    fid = fopen(gt_file);
    switch fscanf(fid, '%c', 1);
        case 'j'
            L = 1;
        case 'n'
            L = 0;
        otherwise
            L = -1;
    end
    fclose(fid);
end

function clearCharactersFromConsole(n)
for i = 1:n
    fprintf('\b')
end
