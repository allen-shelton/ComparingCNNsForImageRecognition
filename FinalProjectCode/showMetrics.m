function showMetrics(numclasses, confusionMatrix, truelabels)
    
    Recall = zeros(numclasses, 1);
    Precision = zeros(numclasses, 1);
    for i = 1:numclasses
        Precision(i) = 100 * (confusionMatrix(i,i) / sum(confusionMatrix(:,i)));
        Recall(i) = 100 * (confusionMatrix(i,i) / sum(confusionMatrix(i,:)));
    end
    F1 = 2 * (Precision .* Recall) ./ (Precision + Recall);
    
    figure
    Metrics = [Recall, Precision, F1];
    bar(categorical(categories(truelabels)), Metrics)
    xlabel('Image Class')
    ylabel('Percentage'), ylim([0 115])
    legend('Recall','Precision','F1 Score')
    title('Recall, Precision, and F1 Score')
    
    figure
    T = table(Recall,Precision,F1,'RowNames',categories(truelabels));
    
    uitable('Data',T{:,:},'ColumnName',T.Properties.VariableNames,...
        'RowName',T.Properties.RowNames,'Units', 'Normalized', 'Position',[0, 0, 1, 1]);