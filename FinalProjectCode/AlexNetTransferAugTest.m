clear, clc, close all

alex = alexnet;
layers = alex.Layers;

inputSize = alex.Layers(1).InputSize;
allImages = imageDatastore('SCUImages', ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');
[trainingImages, valImages] = splitEachLabel(allImages, 0.7, 'randomized');

augimgTrain = augmentedImageDatastore(inputSize(1:2),trainingImages);
augimgValidation = augmentedImageDatastore(inputSize(1:2),valImages);
objectCategories = numel(categories(trainingImages.Labels));

layers(23) = fullyConnectedLayer(objectCategories, ...
    'WeightLearnRateFactor',20, ...
    'BiasLearnRateFactor',20); 
layers(24) = softmaxLayer;
layers(25) = classificationLayer;

%deepNetworkDesigner
%%
options = trainingOptions('adam', ... 
    'InitialLearnRate', 0.00005, ...
    'MaxEpochs', 12, ... 
    'MiniBatchSize', 16, ...
    'ValidationData',augimgValidation, ...
    'ValidationFrequency',2, ...
    'ValidationPatience',Inf, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',4, ...
    'Plots','training-progress', ...
    'Verbose',true, ...
    'VerboseFrequency',20);

[alexTransfer, info1] = trainNetwork(augimgTrain, layers, options);

[predictedLabels,probs] = classify(alexTransfer, augimgValidation); 
[accuracy, loss] = calcAccuracyLoss(predictedLabels, probs, valImages.Labels)

%%
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandYReflection',true);

augimgTrain = augmentedImageDatastore(inputSize(1:2),trainingImages, ...
    'DataAugmentation',imageAugmenter);
augimgValidation = augmentedImageDatastore(inputSize(1:2),valImages);

[alexTransfer, info2] = trainNetwork(augimgTrain, layers, options);

[predictedLabels,probs] = classify(alexTransfer, augimgValidation); 
[accuracy, loss] = calcAccuracyLoss(predictedLabels, probs, valImages.Labels)

%%
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandYReflection',true, ...
    'RandRotation',[-20 20]);

augimgTrain = augmentedImageDatastore(inputSize(1:2),trainingImages, ...
    'DataAugmentation',imageAugmenter);
augimgValidation = augmentedImageDatastore(inputSize(1:2),valImages);

[alexTransfer, info3] = trainNetwork(augimgTrain, layers, options);

[predictedLabels,probs] = classify(alexTransfer, augimgValidation); 
[accuracy, loss] = calcAccuracyLoss(predictedLabels, probs, valImages.Labels)

%%
imageAugmenter = imageDataAugmenter( ...
    'RandXTranslation',[-50 50], ...
    'RandYTranslation',[-50 50]);

augimgTrain = augmentedImageDatastore(inputSize(1:2),trainingImages, ...
    'DataAugmentation',imageAugmenter);
augimgValidation = augmentedImageDatastore(inputSize(1:2),valImages);

[alexTransfer, info4] = trainNetwork(augimgTrain, layers, options);

[predictedLabels,probs] = classify(alexTransfer, augimgValidation); 
[accuracy, loss] = calcAccuracyLoss(predictedLabels, probs, valImages.Labels)

%%
imageAugmenter = imageDataAugmenter( ...
    'RandXShear',[-45 45], ...
    'RandYShear',[-45 45], ...
    'RandScale',[0.5 2]);

augimgTrain = augmentedImageDatastore(inputSize(1:2),trainingImages, ...
    'DataAugmentation',imageAugmenter);
augimgValidation = augmentedImageDatastore(inputSize(1:2),valImages);

[alexTransfer, info5] = trainNetwork(augimgTrain, layers, options);

[predictedLabels,probs] = classify(alexTransfer, augimgValidation); 
[accuracy, loss] = calcAccuracyLoss(predictedLabels, probs, valImages.Labels)

%%
figure
iterations = numel(info1.ValidationAccuracy)/2;
plot(1:iterations,info1.ValidationAccuracy(2:2:end),'LineWidth',2)
hold on
plot(1:iterations,info2.ValidationAccuracy(2:2:end),'LineWidth',2)
plot(1:iterations,info3.ValidationAccuracy(2:2:end),'LineWidth',2)
plot(1:iterations,info4.ValidationAccuracy(2:2:end),'LineWidth',2)
plot(1:iterations,info5.ValidationAccuracy(2:2:end),'LineWidth',2)
title('Validation Accuracy for Different Data Transformations')
xlim([0 80]), xlabel('Iteration')
ylim([0 100]), ylabel('Accuracy (%)')
legend('Case 1 (Base Case)','Case 2','Case 3','Case 4','Case 5')
hold off