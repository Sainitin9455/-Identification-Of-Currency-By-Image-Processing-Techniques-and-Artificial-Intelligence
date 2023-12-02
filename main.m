clc;
close all;
clear ;
%%
[filename,pathname] = uigetfile('*.*','Select the image'); %Open file selection dialog box
img1=imread([pathname,filename]); %Read image from graphics file
imshow(img1), title('Input Image');
%%
resize = imresize(img1,[512 512]);
    matlabroot = cd;    % Dataset path
    datasetpath = fullfile(matlabroot,'Dataset');
    imds = imageDatastore(datasetpath,'IncludeSubfolders',true,'LabelSource','foldernames');
    
    [imdsTrain, imdsValidation] = splitEachLabel(imds,0.7);
    
    augimdsTrain = augmentedImageDatastore([512 512 3],imdsTrain);
    augimdsValidation = augmentedImageDatastore([512 512 3],imdsValidation);
    
    
    layers = [
        imageInputLayer([512 512 3])
        convolution2dLayer(3,8,'Padding','same')
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2,'Stride',2)
        convolution2dLayer(3,16,'Padding','same')
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2,'Stride',2)
        convolution2dLayer(3,32,'Padding','same')
        batchNormalizationLayer
        reluLayer
        fullyConnectedLayer(2)
        softmaxLayer
        classificationLayer];
    
    options = trainingOptions('sgdm', ...
        'InitialLearnRate',0.0001, ...
        'MaxEpochs',20, ...
        'Shuffle','every-epoch', ...
        'ValidationFrequency',10, ...
        'Verbose',false, ...
        'Plots','training-progress');
    
    [net, traininfo] = trainNetwork(augimdsTrain,layers,options);
    
    [YPred,score1] = classify(net,resize);
    accuracy = mean(traininfo.TrainingAccuracy);
    
    msgbox(char(YPred));
    fprintf('Accuracy of classified Model is: %0.4f\n',accuracy);

%%
if strcmp(char(YPred),'Original')
    resize = imresize(img1,[512 512]);
    matlabroot = cd;    % Dataset path
    datasetpath = fullfile(matlabroot,'Denominations');
    imds = imageDatastore(datasetpath,'IncludeSubfolders',true,'LabelSource','foldernames');
    
    [imdsTrain, imdsValidation] = splitEachLabel(imds,0.7);
    
    augimdsTrain = augmentedImageDatastore([512 512 3],imdsTrain);
    augimdsValidation = augmentedImageDatastore([512 512 3],imdsValidation);
    
    
    layers = [
        imageInputLayer([512 512 3])
        convolution2dLayer(3,8,'Padding','same')
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2,'Stride',2)
        convolution2dLayer(3,16,'Padding','same')
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2,'Stride',2)
        convolution2dLayer(3,32,'Padding','same')
        batchNormalizationLayer
        reluLayer
        fullyConnectedLayer(7)
        softmaxLayer
        classificationLayer];
    
    options = trainingOptions('sgdm', ...
        'InitialLearnRate',0.0001, ...
        'MaxEpochs',20, ...
        'Shuffle','every-epoch', ...
        'ValidationFrequency',10, ...
        'Verbose',false, ...
        'Plots','training-progress');
    
    [net, traininfo] = trainNetwork(augimdsTrain,layers,options);
    
    [YPred1,score1] = classify(net,resize);
    accuracy = mean(traininfo.TrainingAccuracy);
    
    msgbox(char(YPred1));
    fprintf('Accuracy of classified Model is: %0.4f\n',accuracy);
end