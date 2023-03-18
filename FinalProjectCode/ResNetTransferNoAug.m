clear, close all

resnet = layerGraph(resnet50);

inputSize = resnet.Layers(1).InputSize;
allImages = imageDatastore('ObjectImages', ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');
[trainingImages, valImages] = splitEachLabel(allImages, 0.7, 'randomized');

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
resnet = replaceLayer(resnet,'fc1000',transferLayers(1));
resnet = replaceLayer(resnet,'fc1000_softmax',transferLayers(2));
resnet = replaceLayer(resnet,'ClassificationLayer_fc1000',transferLayers(3));

%deepNetworkDesigner
%%
options = trainingOptions('adam', ... 
    'InitialLearnRate', 0.00005, ...
    'MaxEpochs', 12, ... 
    'MiniBatchSize', 64, ...
    'ValidationData',augimgValidation, ...
    'ValidationFrequency',2, ...
    'ValidationPatience',10, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',4, ...
    'Plots','training-progress', ...
    'Verbose',true, ...
    'VerboseFrequency',20);

[resnetTransfer, info] = trainNetwork(augimgTrain, resnet, options);

%% Show final Validation Accuracy/Loss
[predictedLabels,probs] = classify(resnetTransfer, augimgValidation); 
[accuracy, loss] = calcAccuracyLoss(predictedLabels, probs, valImages.Labels)

%% Plot Confusion Matrix
confusion = plotConfusionMatrix(predictedLabels, valImages.Labels, accuracy, loss);

%% Shows a random selection of validation images with their predictions
plotPredictions(predictedLabels, probs, valImages);

%% Plot Recall, Precision, and F1 Score data for all classes
showMetrics(objectCategories, confusion, valImages.Labels);

%% Plot ROC Curves for all classes
plotROC(valImages.Labels, probs);