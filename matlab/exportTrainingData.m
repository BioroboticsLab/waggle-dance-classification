function [] = exportTrainingData(source_folder, windowSize, train_proportion)
output_filename = 'train_test.mat';
folder_obj  = dir(source_folder);
nDirs       = length(folder_obj);
progress    = 0;
X = [];
Y = [];
q = [];
save(output_filename, 'X', 'Y', 'q', '-v7.3');
matfile_obj = matfile(output_filename, 'Writable', true);
nBatchExport = 100;
nCount = nBatchExport;

fprintf('\n\n\n')

% Traverse folder structure
% first level corresponds to hour-minute-cam_id
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
                imgfolderPath = fullfile(subpath, f.name);
                
                % load label from ground truth file
                L = loadGroundTruth(imgfolderPath);
                
                % go to next folder if no ground truth could be found
                if L < 0 
                    continue;
                end
                
                ImageTensor = loadImages(imgfolderPath);
                
                % prepare data structures for current waggle run
                [dX] = createSlidingWindowTensor(ImageTensor, windowSize);
                dY = repmat(L, size(dX, 1), 1);

                X = [X; dX];
                Y = [Y; dY];
                
                if nCount > 0
                    nCount = nCount-1;
                else
                    % concatenate with previously stored data
                    if isempty(matfile_obj.q)
                        matfile_obj.X = X;
                        matfile_obj.Y = Y;
                        matfile_obj.q = 1;
                    else
                        deltaSize   = size(Y, 1);
                        curSize     = size(matfile_obj, 'Y');
                        idxNew      = curSize+1 : curSize + deltaSize;
                        matfile_obj.X(idxNew, :, :, :) = X;
                        matfile_obj.Y(idxNew, 1) = Y;
                        X = [];
                        Y = [];
                        nCount = nBatchExport;
                    end
                end
                    
                
            end
        end
    end
end

nData       = size(matfile_obj, 'Y', 1);
idxSplit    = floor(train_proportion * nData);
perm        = randperm(nData);
j = perm(1);
matfile_obj.X_test = matfile_obj.X(j, :, :, :);
matfile_obj.Y_test = matfile_obj.Y(j,1);
matfile_obj.X_train = matfile_obj.X(j, :, :, :);
matfile_obj.Y_train = matfile_obj.Y(j,1);

% shuffle and split in train and test set

for i = 2 : nData
    j = perm(i);
    
    if i > idxSplit
        matfile_obj.X_test(i, :, :, :)  = matfile_obj.X(j, :, :, :);
        matfile_obj.Y_test(i, 1)           = matfile_obj.Y(j, 1);
    else
        matfile_obj.X_train(i, :, :, :)  = matfile_obj.X(j, :, :, :);
        matfile_obj.Y_train(i, 1)           = matfile_obj.Y(j, 1);
    end
end



% save(output_filename, 'X_train', 'Y_train', 'X_test', 'Y_test');


% this function takes a sequence of images and creates a 4D tensor
% of subsequences of length <window_size> images
function M = createSlidingWindowTensor(ImageTensor, windowSize)
[nFrames, imgDim, ~] = size(ImageTensor);
nWindows = (nFrames - windowSize) + 1;
M = uint8(zeros(nWindows, windowSize, imgDim, imgDim));
for k = 1 : nWindows 
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
