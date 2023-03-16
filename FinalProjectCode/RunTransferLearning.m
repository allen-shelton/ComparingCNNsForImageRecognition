
prompt = ['For which neural network would you like to perform transfer learning?' ...
        '\n1) AlexNet' ...
        '\n2) VGG-16' ...
        '\n3) GoogleNet' ...
        '\n4) Resnet50' ...
        '\n5) Exit Program' ...
        '\nType the corresponding number: '];
while(1)
    network  = input(prompt);
    switch network
        case 1
            disp('Training AlexNet...');
            break
        case 2
            disp('Training VGG-16...');
            break
        case 3
            disp('Training GoogleNet...');
            break
        case 4
            disp('Training ResNet50...');
            break
        case 5
            disp('Exiting Program...');
            break
        otherwise
            disp('Invalid Option. Restarting...');
            pause(1.5);
    end
end

