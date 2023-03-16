clear, clc, close all

google = layerGraph(googlenet);

inputSize = google.Layers(1).InputSize;
allImages = imageDatastore('SCUImages', ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');
[trainingImages, valImages] = splitEachLabel(allImages, 0.7, 'randomized');
% imageAugmenter = imageDataAugmenter( ...
%     'RandRotation',[-20,20], ...
%     'RandXTranslation',[-3 3], ...
%     'RandYTranslation',[-3 3]);
augimgTrain = augmentedImageDatastore(inputSize(1:2),trainingImages);
augimgValidation = augmentedImageDatastore(inputSize(1:2),valImages);
objectCategories = numel(categories(trainingImages.Labels));

transferLayers = [
    fullyConnectedLayer(objectCategories, ...
    'WeightLearnRateFactor',20, ...
    'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer
    ];
google = replaceLayer(google,'loss3-classifier',transferLayers(1));
google = replaceLayer(google,'prob',transferLayers(2));
google = replaceLayer(google,'output',transferLayers(3));

%deepNetworkDesigner
%%
options = trainingOptions('adam', ... 
    'InitialLearnRate', 0.00005, ...
    'MaxEpochs', 15, ... 
    'MiniBatchSize', 16, ...
    'ValidationData',augimgValidation, ...
    'ValidationFrequency',4, ...
    'ValidationPatience',10, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',4, ...
    'Plots','training-progress', ...
    'Verbose',true, ...
    'VerboseFrequency',20);

[googleTransfer, info] = trainNetwork(augimgTrain, google, options);

%%
[predictedLabels,probs] = classify(googleTransfer, augimgValidation); 
[accuracy, loss] = calcAccuracyLoss(predictedLabels, probs, valImages.Labels)

%%
confusion = plotConfusionMatrix(predictedLabels, valImages.Labels, accuracy, loss);

%%
plotPredictions(predictedLabels, probs, valImages);

%%
showMetrics(objectCategories, confusion, valImages.Labels);

%%
plotROC(valImages.Labels, probs);