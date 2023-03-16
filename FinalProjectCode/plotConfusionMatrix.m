function confusion = plotConfusionMatrix(predictions, truelabels, accuracy, loss)
    figure
    confusion = confusionmat(truelabels,predictions);
    confusionchart(confusion)
    title("Confusion Matrix; Accuracy = "+num2str(accuracy)+"; Loss = "+num2str(loss))