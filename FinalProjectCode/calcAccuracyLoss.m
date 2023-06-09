function [accuracy, loss] = calcAccuracyLoss(predictions, probabilities, truelabels)
    % Calculates the final validation accuracy and loss of the retrained network
    
    accuracy = mean(predictions == truelabels);

    onehotLabels = onehotencode(truelabels,2);
    mask = probabilities .* onehotLabels;
    logmask = log(mask);
    logmask(logmask == -Inf) = 0;
    loss = -(1/numel(truelabels)) * sum(logmask,'all');