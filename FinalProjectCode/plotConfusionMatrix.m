function confusion = plotConfusionMatrix(predictions, truelabels, accuracy, loss)
    % This function plots the confusion matrix for the validation data
    figure
    confusion = confusionmat(truelabels,predictions);
    confusionchart(confusion)
    title("Confusion Matrix; Accuracy = "+num2str(accuracy)+"; Loss = "+num2str(loss))