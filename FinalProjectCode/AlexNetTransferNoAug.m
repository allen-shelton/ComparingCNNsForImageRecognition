clear, close all
%% Use AlexNet for Transfer Learning %%

% Load AlexNet pretrained model
alex = alexnet;
layers = alex.Layers;

% Load image data
inputSize = alex.Layers(1).InputSize;
allImages = imageDatastore('ObjectImages', ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');
[trainingImages, valImages] = splitEachLabel(allImages, 0.7, 'randomized');

% Resize images to 224 x 224
augimgTrain = augmentedImageDatastore(inputSize(1:2),trainingImages, ...
    'DataAugmentation',imageAugmenter);
augimgValidation = augmentedImageDatastore(inputSize(1:2),valImages);
objectCategories = numel(categories(trainingImages.Labels));

% Reconfigure final layers to fit the size of our dataset for transfer learning
layers(23) = fullyConnectedLayer(objectCategories, ...
    'WeightLearnRateFactor',20, ...
    'BiasLearnRateFactor',20); 
layers(24) = softmaxLayer;
layers(25) = classificationLayer;

%deepNetworkDesigner

% Set training options
options = trainingOptions('adam', ... 
    'InitialLearnRate', 0.00005, ...
    'MaxEpochs', 12, ... 
    'MiniBatchSize', 16, ...
    'ValidationData',augimgValidation, ...
    'ValidationFrequency',2, ...
    'ValidationPatience',10, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',4, ...
    'Plots','training-progress', ...
    'Verbose',true, ...
    'VerboseFrequency',20);

% Run transfer learning with training data and modified network
alexTransfer = trainNetwork(augimgTrain, layers, options);

%% Show final Validation Accuracy/Loss
[predictedLabels,probs] = classify(alexTransfer, augimgValidation); 
[accuracy, loss] = calcAccuracyLoss(predictedLabels, probs, valImages.Labels)

%% Plot Confusion Matrix
confusion = plotConfusionMatrix(predictedLabels, valImages.Labels, accuracy, loss);

%% Shows a random selection of validation images with their predictions
plotPredictions(predictedLabels, probs, valImages);

%% Plot Recall, Precision, and F1 Score data for all classes
showMetrics(objectCategories, confusion, valImages.Labels);

%% Plot ROC Curves for all classes
plotROC(valImages.Labels, probs);