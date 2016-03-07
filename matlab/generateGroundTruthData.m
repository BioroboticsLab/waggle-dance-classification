% Allows user to tell if bees are doing the waggle-dance on a series of images in a
% WDD-output structure
% Arguments: 
%   folderPath : set to the Folder where all the dances can be found
% To navigate through waggle dance folders use up and down
% To navigate through pictures use left and right.
% To mark the folder use:
% j: Mark the pictures as a dance
% n: Mark the pictures as not a dance
% v: Mark the pictures as not marked
% esc: close program
% When changing folders a csv gets saved, with the given mark.
% If no mark was given it gets the mark v.
function classifyDance(folderPath)
folder = dir(folderPath);
i = 0;
UD.currentImgFolder = 1;
UD.imgFolders = {};
UD.foundsub = 0;
%traverse folder structure
for d = folder'
    if d.isdir && ~strcmp(d.name,'.') && ~strcmp(d.name,'..')
        UD.foundsub = 1;
        subpath = fullfile(folderPath,d.name);
        subfolder = dir(subpath);
        for f = subfolder'
            if f.isdir && ~strcmp(f.name,'.') && ~strcmp(f.name,'..')
                imgfolderpath = fullfile(subpath, f.name);
                UD.imgFolders{end+1} = imgfolderpath;
                if i == 0
                    UD.imgNumber    = 1;
                    UD.imageArray   = dir(fullfile(imgfolderpath,'*.png'));
                    UD.img          = imread(fullfile(imgfolderpath,UD.imageArray(UD.imgNumber).name));
                    UD.img          = imresize(UD.img, 4.0);
                end
                i = i + 1;
            end
        end
    end
end

if ~UD.foundsub
    UD.imgNumber    = 1;
    UD.imageArray   = dir(fullfile(folderPath,'*.png'));
    UD.img          = imread(fullfile(folderPath,UD.imageArray(UD.imgNumber).name));
    UD.img          = imresize(UD.img, 4.0);
end
% tag for bee dance confirmation j=confirmed n=no dance v=unsure if it's a
% dance
UD.conf         = 'v';
UD.fig          = figure(1);
length = size(UD.imageArray);
s = sprintf('Image %d of %d', UD.imgNumber, length(1:1));
length = size(UD.imgFolders);
if UD.foundsub
    s = sprintf('%s | Folder %d of %d',s , UD.currentImgFolder, length(2:2));
end
set(UD.fig, 'Name', s);

imshow(UD.img);
hold on;
UD.clicked      = 0;

set(UD.fig, 'UserData',              UD);
set(UD.fig, 'KeyPressFcn',           @keyPressed);
set(UD.fig, 'Interruptible',         'off');
if UD.foundsub
    s = sprintf('Current folder: %s', UD.imgFolders{UD.currentImgFolder});
else
    s = sprintf('Current folder: %s', folderPath);
end
title(s);
waitfor(UD.fig)
% eval('waitfor(UD.fig)')

close all

% check for different functions when key is pressend
function keyPressed(~, eventdata)
UD = get(gcf, 'UserData');
eventdata.Key;
switch eventdata.Key
    % it is a dance
    case 'j'
        UD.conf = 'j';
        disp('Dance detected');
    %it is no dance
    case 'n'
        UD.conf = 'n';
        disp('No dance detected');
    % it might be a dance
    case 'v'
        UD.conf = 'v';
        disp('Maybe the bee is dancing, who knows');
    % close program
    case 'escape'        
        f               = gcf;
        close(f);
        waitfor(f) 
    % next image
    case 'rightarrow'
        array = size(UD.imageArray);
        if UD.imgNumber + 1 < array(1:1)
            cla;
            UD.imgNumber    = UD.imgNumber+1;
            length = size(UD.imageArray);
            s = sprintf('Image %d of %d', UD.imgNumber, length(1:1));
            length = size(UD.imgFolders);
            if UD.foundsub
                s = sprintf('%s | Folder %d of %d',s , UD.currentImgFolder, length(2:2));
            end
            set(UD.fig, 'Name', s);
            if UD.foundsub
                UD.img          = imread(fullfile(UD.imgFolders{UD.currentImgFolder},UD.imageArray(UD.imgNumber).name));
            else
                UD.img          = imread(fullfile(folderPath,UD.imageArray(UD.imgNumber).name));
            end
            UD.img          = imresize(UD.img, 4.0);
            imshow(UD.img);
            drawnow;
            UD.clicked      = 0;
        end
    % former image
    case 'leftarrow'
        if UD.imgNumber - 1 > 0
            cla;
            UD.imgNumber    = UD.imgNumber-1;
            length = size(UD.imageArray);
            s = sprintf('Image %d of %d', UD.imgNumber, length(1:1));
            length = size(UD.imgFolders);
            if UD.foundsub
                s = sprintf('%s | Folder %d of %d',s , UD.currentImgFolder, length(2:2));
            end
            set(UD.fig, 'Name', s);
            if UD.foundsub
                UD.img          = imread(fullfile(UD.imgFolders{UD.currentImgFolder},UD.imageArray(UD.imgNumber).name));
            else
                UD.img          = imread(fullfile(folderPath,UD.imageArray(UD.imgNumber).name));
            end
            UD.img          = imresize(UD.img, 4.0);
            imshow(UD.img);
            drawnow;
            UD.clicked      = 0;
        end
    % next folder
    case 'uparrow'
        length = size(UD.imgFolders);
        if UD.currentImgFolder + 1 < length(2:2)
            if UD.foundsub
                path = UD.imgFolders{UD.currentImgFolder};
                OUT = cell(1);
                OUT{1,1} = UD.conf;
                cell2csv(fullfile(path,'gt.csv'), OUT);
                s = sprintf('Saved file to %s', fullfile(path,'gt.csv'));
            else
                path = folderPath;
                OUT = cell(1);
                OUT{1,1} = UD.conf;
                cell2csv(fullfile(path,'gt.csv'), OUT);
                s = sprintf('Saved file to %s', fullfile(path,'gt.csv'));
            end
            disp(s)
            UD.currentImgFolder = UD.currentImgFolder + 1;
            UD.conf = 'v';
            UD.imgNumber = 1;
            UD.imageArray   = dir(fullfile(UD.imgFolders{UD.currentImgFolder},'*.png'));
            disp(fullfile(UD.imgFolders{UD.currentImgFolder}));
            UD.img          = imread(fullfile(UD.imgFolders{UD.currentImgFolder},UD.imageArray(UD.imgNumber).name));
            UD.img          = imresize(UD.img, 4.0);
            imshow(UD.img);
            drawnow;
            UD.clicked      = 0;
            s = sprintf('Current folder: %s', UD.imgFolders{UD.currentImgFolder});
            title(s);
            length = size(UD.imageArray);
            s = sprintf('Image %d of %d', UD.imgNumber, length(1:1));
            length = size(UD.imgFolders);
            if UD.foundsub
                s = sprintf('%s | Folder %d of %d',s , UD.currentImgFolder, length(2:2));
            end
            set(UD.fig, 'Name', s);
            s = sprintf('Loaded images from path %s', UD.imgFolders{UD.currentImgFolder});
            disp(s);
        end
    % former folder
    case 'downarrow'
        if UD.currentImgFolder - 1 > 0
            if UD.foundsub
                path = UD.imgFolders{UD.currentImgFolder};
                OUT = cell(1);
                OUT{1,1} = UD.conf;
                cell2csv(fullfile(path,'gt.csv'), OUT);
                s = sprintf('Saved file to %s', fullfile(path,'gt.csv'));
            else
                path = folderPath;
                OUT = cell([1]);
                OUT{1,1} = UD.conf;
                cell2csv(fullfile(path,'gt.csv'), OUT);
                s = sprintf('Saved file to %s', fullfile(path,'gt.csv'));
            end
            disp(s)
            UD.currentImgFolder = UD.currentImgFolder - 1;
            UD.conf = 'v';
            UD.imgNumber = 1;
            UD.imageArray   = dir(fullfile(UD.imgFolders{UD.currentImgFolder},'*.png'));
            disp(fullfile(UD.imgFolders{UD.currentImgFolder}));
            UD.img          = imread(fullfile(UD.imgFolders{UD.currentImgFolder},UD.imageArray(UD.imgNumber).name));
            UD.img          = imresize(UD.img, 4.0);
            imshow(UD.img);
            drawnow;
            UD.clicked      = 0;
            s = sprintf('Current folder: %s', UD.imgFolders{UD.currentImgFolder});
            title(s);
            length = size(UD.imageArray);
            s = sprintf('Image %d of %d', UD.imgNumber, length(1:1));
            length = size(UD.imgFolders);
            if UD.foundsub
                s = sprintf('%s | Folder %d of %d',s , UD.currentImgFolder, length(2:2));
            end
            set(UD.fig, 'Name', s);
            s = sprintf('Loaded images from path %s', UD.imgFolders{UD.currentImgFolder});
            disp(s);
        end
    otherwise
end
set(gcbo, 'UserData', UD);               %write user data back to figure