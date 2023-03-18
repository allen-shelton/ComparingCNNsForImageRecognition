function plotPredictions(predictions, probabilities, images)
    % This function displays 16 random validation images along with their
    % prediction and probability score
    
    idx = randperm(numel(images.Files),16);
    figure
    for i = 1:16
        subplot(4,4,i)
        I = readimage(images,idx(i));
        imshow(I)
        label = predictions(idx(i));
        title(string(label) + ", " + num2str(100*max(probabilities(idx(i),:)),3) + "%");
    end