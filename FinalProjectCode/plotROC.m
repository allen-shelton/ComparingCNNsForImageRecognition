function plotROC(truelabels, probabilities)
    fig = figure;
    Labels = categories(truelabels);
    for i = 1:numel(Labels)
        subplot(3,4,i)
        OvAprobs = probabilities(:,i) - max(probabilities(:,1:end ~= i),[],2);
        [X,Y,~,AUC,optrocpt] = perfcurve(truelabels,OvAprobs,Labels(i));
        if AUC > 0.95
            color = 'b';
        else
            color = 'r';
        end
        plot(X,Y,color)
        hold on
        plot(optrocpt(1),optrocpt(2),'ro');
        title(num2str(Labels(i)+"; AUC = "+num2str(AUC)))
        ylim([0 1.1])
        xlim([-0.02 1])
        legend('','Optimal Operating Point')
        legend('boxoff')
        hold off
    end
    han=axes(fig,'visible','off'); 
    han.Title.Visible='on';
    han.XLabel.Visible='on';
    han.YLabel.Visible='on';
    ylabel(han,'True Positive Rate');
    xlabel(han,'False Negative Rate');
    %title(han,'ROC Curve for All Classes');