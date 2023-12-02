clear all;close all;clc;
srcFiles = dir('Z:\Worked\Identification of Currency via image recognition\CODE\Sample Datset\*.*');  % the folder in which ur images exists
for i = 3 : length(srcFiles)
filename = strcat('Z:\Worked\Identification of Currency via image recognition\CODE\Sample Datset\',srcFiles(i).name);
im = imread(filename);
im = imresize(im,[260 540]);
newfilename=fullfile('Z:\Worked\Identification of Currency via image recognition\CODE\Dataset\Original\',['Original' num2str(i-2) '.png']);
imwrite(im,newfilename,'png');
end