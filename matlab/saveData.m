% Das Programm geht einen Tag einer WDD-Ordnerstruktur durch und speichert
% für jeden Tanz folgende Informationen in einer CSV-Datei: #Frames Orientation PosX PosY TIME
% Der Ort der WDD-Ordnerstruktur muss statisch im Programmcode angegeben
% werden. Der Name Ausgabedatei ist ebenfalls statisch im Code und heißt
% tag.csv
clearvars
folderPath='/Users/mehmedhalilovic/Documents/MATLAB/BA/PlotWDDdataOnMap/20140822CleanFinish';
folder = dir(folderPath);
UD.currentImgFolder = 1;
UD.imgFolders = {};
progress = 0; %number of dances that have been processed 
csvdata = [];
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
                UD.csvArray     = dir(fullfile(imgfolderpath,'*.csv'));
                numberOfImages = length(UD.imageArray);
                progress = progress + 1
                %If there is no GroundTruth for this data it gets an error value
                for csvname = UD.csvArray
                    fid = fopen(fullfile(imgfolderpath, csvname.name),'r');
                    T = textscan(fid, repmat('%s',1,10), 'delimiter',' ', 'CollectOutput',true);
                    T = T{1};
                    csvdata = [csvdata; T(2,2) T(1,3) T(1,1) T(1,2) T(2,1)];
                    fclose(fid);
                end
            end
        end
    end
end
c = csvdata; 
 fid = fopen('newdata.csv', 'w') ;
 for i = 1:length(c)
    fprintf(fid, '%s,', c{i,1:end-1}) ;
    fprintf(fid, '%s ', '2014/08/22') ;
    fprintf(fid, '%s\n', c{i,end}) ; 
 end
 fclose(fid) ; 
 
Data = fileread('newdata.csv');
Data = strrep(Data, '$', '0');
FID = fopen('tag.csv', 'w');
fwrite(FID, Data, 'char');
fclose(FID);
