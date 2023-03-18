clear, close all

vgg = vgg16;
layers = vgg.Layers;

inputSize = vgg.Layers(1).InputSize;
allImages = imageDatastore('ObjectImages', ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');
[trainingImages, valImages] = splitEachLabel(allImages, 0.7, 'randomized');

augimgTrain = augmentedImageDatastore(inputSize(1:2),trainingImages);
augimgValidation = augmentedImageDatastore(inputSize(1:2),valImages);
objectCategories = numel(categories(trainingImages.Labels));

layers(39) = fullyConnectedLayer(objectCategories, ...
    'WeightLearnRateFactor',20, ...
    'BiasLearnRateFactor',20); 
layers(40) = softmaxLayer;
layers(41) = classificationLayer;

%deepNetworkDesigner

% Set training options
options = trainingOptions('adam', ... 
    'InitialLearnRate', 0.00005, ...
    'MaxEpochs', 12, ... 
    'MiniBatchSize', 32, ...
    'ValidationData',augimgValidation, ...
    'ValidationFrequency',4, ...
    'ValidationPatience',10, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',4, ...
    'Plots','training-progress', ...
    'Verbose',true, ...
    'VerboseFrequency',20);

% Run transfer learning
vggTransfer = trainNetwork(augimgTrain, layers, options);

%% Show final Validation Accuracy/Loss
[predictedLabels,probs] = classify(vggTransfer, augimgValidation); 
[accuracy, loss] = calcAccuracyLoss(predictedLabels, probs, valImages.Labels)

%% Plot Confusion Matrix
confusion = plotConfusionMatrix(predictedLabels, valImages.Labels, accuracy, loss);

%% Shows a random selection of validation images with their predictions
plotPredictions(predictedLabels, probs, valImages);

%% Plot Recall, Precision, and F1 Score data for all classes
showMetrics(objectCategories, confusion, valImages.Labels);

%% Plot ROC Curves for all classes
plotROC(valImages.Labels, probs);