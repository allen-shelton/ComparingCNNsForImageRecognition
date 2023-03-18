clc, clear, close all
%% Run Any Test Case %%
% This script allows you to easily choose which neural network you want to
% train and whether or not you want to use data augmentation. You're
% welcome to go into any of the other scripts separately and play around
% with the different parameters to see if you can get better results. By
% default all script in the project are configured to train the Group 1
% dataset, or the "ObjectImages" folder. 

prompt = ['For which neural network would you like to perform transfer learning?' ...
        '\n1) AlexNet' ...
        '\n2) VGG-16' ...
        '\n3) GoogleNet' ...
        '\n4) Resnet50' ...
        '\n5) Data Augmentation Test with AlexNet' ...
        '\n6) Exit Program' ...
        '\nType the corresponding number: '];

while(1)
    network  = input(prompt);
    switch network
        case 1
            aug = input('Would you like to use Data Augmentation [y/n]: ','s');
            if aug == 'y'
                disp('Training AlexNet with Data Augmentation...');
                AlexNetTransfer
                break
            elseif aug == 'n'
                disp('Training AlexNet without Data Augmentation...');
                AlexNetTransferNoAug
                break
            else
                disp('Invalid Option. Restarting...');
                pause(1.5);
            end
        case 2
            aug = input('Would you like to use Data Augmentation [y/n]: ','s');
            if aug == 'y'
                disp('Training VGG-16 with Data Augmentation...');
                VGGTransfer
                break
            elseif aug == 'n'
                disp('Training VGG-16 without Data Augmentation...');
                VGGTransferNoAug
                break
            else
                disp('Invalid Option. Restarting...');
                pause(1.5);
            end
        case 3
            aug = input('Would you like to use Data Augmentation [y/n]: ','s');
            if aug == 'y'
                disp('Training GoogleNet with Data Augmentation...');
                GoogleNetTransfer
                break
            elseif aug == 'n'
                disp('Training GoogleNet without Data Augmentation...');
                GoogleNetTransferNoAug
                break
            else
                disp('Invalid Option. Restarting...');
                pause(1.5);
            end
        case 4
            aug = input('Would you like to use Data Augmentation [y/n]: ','s');
            if aug == 'y'
                disp('Training ResNet50 with Data Augmentation...');
                ResNetTransfer
                break
            elseif aug == 'n'
                disp('Training ResNet50 without Data Augmentation...');
                ResNetTransferNoAug
                break
            else
                disp('Invalid Option. Restarting...');
                pause(1.5);
            end
        case 5
            disp('Running Data Augmentation Test with AlexNet...');
            break
        case 6
            disp('Exiting Program...');
            break
        otherwise
            disp('Invalid Option. Restarting...');
            pause(1.5);
    end
end

