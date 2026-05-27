%% train_cnn.m
% Deep Learning CNN classification in MATLAB using Deep Learning Toolbox
% Huấn luyện mô hình mạng nơ-ron tích chập (CNN) trực tiếp trên ảnh gốc.

warning('off', 'all');
fprintf('=== TRAINING DEEP LEARNING CNN IN MATLAB ===\n');

% Set up paths
script_dir = fileparts(mfilename('fullpath'));
dataset_dir = fullfile(script_dir, '..', 'data', 'segmented', 'dataset_segmented');
models_dir = fullfile(script_dir, '..', 'models');
reports_dir = fullfile(script_dir, '..', 'reports');

if ~exist(reports_dir, 'dir')
    mkdir(reports_dir);
end

% 1. Create Image Datastore
if ~exist(dataset_dir, 'dir')
    error('Thu muc dataset_segmented khong ton tai. Vui long kiem tra lai.');
end
imds = imageDatastore(dataset_dir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Get info about classes
class_names = categories(imds.Labels);
num_classes = numel(class_names);
fprintf('Total images found: %d across %d classes.\n', numel(imds.Files), num_classes);

% 2. Split Data (Balanced subsets for CPU speed)
% 50 images per class for training, 50 images per class for testing/validation
num_train_per_class = 50;
num_val_per_class = 50;
[imdsTrain, imdsVal] = splitEachLabel(imds, num_train_per_class, num_val_per_class, 'randomize');

fprintf('Training set size: %d, Validation set size: %d\n', numel(imdsTrain.Files), numel(imdsVal.Files));

% 3. Image Augmenter (Resizing on-the-fly to 128x128x3)
input_size = [128 128 3];
imdsTrainAug = augmentedImageDatastore(input_size, imdsTrain);
imdsValAug = augmentedImageDatastore(input_size, imdsVal);

% 4. Define Network Architecture
layers = [
    imageInputLayer(input_size, 'Name', 'input')
    
    convolution2dLayer(3, 8, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool1')
    
    convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool2')
    
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool3')
    
    fullyConnectedLayer(num_classes, 'Name', 'fc')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classoutput')
];

% 5. Training Options
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 8, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', imdsValAug, ...
    'ValidationFrequency', 20, ...
    'Verbose', true, ...
    'Plots', 'none', ...
    'ExecutionEnvironment', 'cpu');

% 6. Train network
fprintf('\n--- Training custom CNN on CPU (this should take ~1-2 minutes) ---\n');
tic;
net = trainNetwork(imdsTrainAug, layers, options);
time_cnn_train = toc;
fprintf('CNN trained successfully in %.2f seconds.\n', time_cnn_train);

% 7. Evaluate on Validation Set
tic;
[YPred, probs] = classify(net, imdsValAug);
time_cnn_inference = toc;

Y_val = imdsVal.Labels;
C = confusionmat(Y_val, YPred);

% Calculate Metrics
acc_cnn = sum(diag(C)) / sum(C(:));

precisions_cnn = zeros(num_classes, 1);
recalls_cnn = zeros(num_classes, 1);
f1s_cnn = zeros(num_classes, 1);

for c = 1:num_classes
    tp = C(c, c);
    fp = sum(C(:, c)) - tp;
    fn = sum(C(c, :)) - tp;
    
    if (tp + fp) > 0
        precisions_cnn(c) = tp / (tp + fp);
    else
        precisions_cnn(c) = 0;
    end
    
    if (tp + fn) > 0
        recalls_cnn(c) = tp / (tp + fn);
    else
        recalls_cnn(c) = 0;
    end
    
    if (precisions_cnn(c) + recalls_cnn(c)) > 0
        f1s_cnn(c) = 2 * (precisions_cnn(c) * recalls_cnn(c)) / (precisions_cnn(c) + recalls_cnn(c));
    else
        f1s_cnn(c) = 0;
    end
end

macro_f1_cnn = mean(f1s_cnn);

fprintf('\nCNN Model Evaluation:\n');
fprintf('  - Accuracy: %.4f\n', acc_cnn);
fprintf('  - Macro F1-Score: %.4f\n', macro_f1_cnn);
fprintf('  - Inference Time for Validation Set: %.4f seconds\n', time_cnn_inference);

% Save model
cnn_model_path = fullfile(models_dir, 'cnn_model.mat');
save(cnn_model_path, 'net', 'time_cnn_train', 'acc_cnn', 'macro_f1_cnn');
fprintf('Saved trained CNN model to %s\n', cnn_model_path);

% Plot and save confusion matrix
figure('Visible', 'off');
cm = confusionchart(Y_val, YPred);
cm.Title = 'Confusion Matrix: Custom CNN (Raw Images)';
saveas(gcf, fullfile(reports_dir, 'confusion_matrix_cnn.png'));
close(gcf);
fprintf('Saved confusion matrix plot to reports/confusion_matrix_cnn.png\n');

% Save evaluation variables for comparison script
save(fullfile(reports_dir, 'cnn_results.mat'), 'Y_val', 'YPred', 'time_cnn_train', 'time_cnn_inference', 'acc_cnn', 'macro_f1_cnn');

fprintf('CNN Training and Evaluation Completed!\n');
