%% train_morphology_ml.m
% Fused Feature Machine Learning (Morphology + Color HSV + Texture GLCM)
% Huấn luyện mô hình học máy truyền thống sử dụng đặc trưng dung hợp (Hình thái + Màu sắc + Kết cấu).

warning('off', 'all');
fprintf('=== TRAINING TRADITIONAL ML MODELS ON FUSED FEATURES ===\n');

% Set up paths
script_dir = fileparts(mfilename('fullpath'));
morph_csv_path = fullfile(script_dir, '..', 'data', 'processed', 'dataset_morphology.csv');
texture_csv_path = fullfile(script_dir, '..', 'data', 'processed', 'dataset_texture_color.csv');
models_dir = fullfile(script_dir, '..', 'models');
reports_dir = fullfile(script_dir, '..', 'reports');

if ~exist(reports_dir, 'dir')
    mkdir(reports_dir);
end

% 1. Load Morphological Dataset
if ~exist(morph_csv_path, 'file')
    error('File dataset_morphology.csv khong ton tai.');
end
data_morph = readtable(morph_csv_path);

% 2. Load Color and Texture Dataset
if ~exist(texture_csv_path, 'file')
    error('File dataset_texture_color.csv khong ton tai. Vui long chay GLCM_HSV_Extraction.m truoc.');
end
data_texture = readtable(texture_csv_path);

% Check alignment and row counts
if height(data_morph) ~= height(data_texture)
    error('So luong dong giua dataset_morphology (%d) va dataset_texture_color (%d) khong khop!', ...
        height(data_morph), height(data_texture));
end

% Verify labels match
labels_match = all(strcmp(string(data_morph.Label), string(data_texture.Label)));
if ~labels_match
    fprintf('Warning: Nhan cua hai dataset khong khop hoan toan tai tung dong. Tien hanh ghep theo index va dung nhan tu file morphology.\n');
end

% Combine features
% Morphology features (columns 2:end)
X_morph = data_morph{:, 2:end};
% Color & Texture features (columns 2:end)
X_texture = data_texture{:, 2:end};

% Concatenate horizontally
X = [X_morph, X_texture];

% Handle NaN values (e.g., in GLCM_Correlation for uniform regions)
X = fillmissing(X, 'constant', 0);

Y = categorical(data_morph.Label);

class_names = categories(Y);
num_classes = numel(class_names);
fprintf('Fused Dataset contains %d samples, %d features (%d Morph + %d Color/Texture), %d classes.\n', ...
    size(X, 1), size(X, 2), size(X_morph, 2), size(X_texture, 2), num_classes);

% 3. Split Data (80% Train, 20% Test) - Stratified Split
rng(42); % For reproducibility
cv = cvpartition(Y, 'Holdout', 0.2);
X_train = X(training(cv), :);
Y_train = Y(training(cv));
X_test = X(test(cv), :);
Y_test = Y(test(cv));

fprintf('Train size: %d, Test size: %d\n', size(X_train, 1), size(X_test, 1));

% 4. Standardize Features (Z-Score Normalization)
mu = mean(X_train);
sigma = std(X_train);
sigma(sigma == 0) = 1; % Prevent division by zero

X_train_scaled = (X_train - mu) ./ sigma;
X_test_scaled = (X_test - mu) ./ sigma;

% 5. Train Fused ML Models
fprintf('\n--- 1. Training Fused Support Vector Machine (SVM - RBF Kernel) ---\n');
tic;
t_svm = templateSVM('KernelFunction', 'rbf', 'Standardize', false);
svm_model = fitcecoc(X_train_scaled, Y_train, 'Learners', t_svm);
time_svm_train = toc;
fprintf('SVM trained in %.2f seconds.\n', time_svm_train);

fprintf('\n--- 2. Training Fused Random Forest ---\n');
tic;
t_tree = templateTree('MaxNumSplits', 100);
rf_model = fitcensemble(X_train_scaled, Y_train, 'Method', 'Bag', ...
    'NumLearningCycles', 100, 'Learners', t_tree);
time_rf_train = toc;
fprintf('Random Forest trained in %.2f seconds.\n', time_rf_train);

fprintf('\n--- 3. Training Fused K-Nearest Neighbors (KNN, K=5) ---\n');
tic;
knn_model = fitcknn(X_train_scaled, Y_train, 'NumNeighbors', 5);
time_knn_train = toc;
fprintf('KNN trained in %.2f seconds.\n', time_knn_train);

% 6. Evaluate Models on Test Set
models = {svm_model, rf_model, knn_model};
model_names = {'SVM (RBF)', 'Random Forest', 'KNN (K=5)'};
f1_scores = zeros(1, 3);
accuracies = zeros(1, 3);
predictions = cell(1, 3);

for m = 1:numel(models)
    tic;
    pred = predict(models{m}, X_test_scaled);
    time_inference = toc;
    predictions{m} = pred;
    
    % Confusion Matrix
    C = confusionmat(Y_test, pred);
    
    % Metrics Calculation
    acc = sum(diag(C)) / sum(C(:));
    accuracies(m) = acc;
    
    precisions = zeros(num_classes, 1);
    recalls = zeros(num_classes, 1);
    f1s = zeros(num_classes, 1);
    
    for c = 1:num_classes
        tp = C(c, c);
        fp = sum(C(:, c)) - tp;
        fn = sum(C(c, :)) - tp;
        
        if (tp + fp) > 0
            precisions(c) = tp / (tp + fp);
        else
            precisions(c) = 0;
        end
        
        if (tp + fn) > 0
            recalls(c) = tp / (tp + fn);
        else
            recalls(c) = 0;
        end
        
        if (precisions(c) + recalls(c)) > 0
            f1s(c) = 2 * (precisions(c) * recalls(c)) / (precisions(c) + recalls(c));
        else
            f1s(c) = 0;
        end
    end
    
    % Macro Average
    macro_f1 = mean(f1s);
    f1_scores(m) = macro_f1;
    
    fprintf('%s Model:\n', model_names{m});
    fprintf('  - Accuracy: %.4f\n', acc);
    fprintf('  - Macro F1-Score: %.4f\n', macro_f1);
    fprintf('  - Inference Time for Test Set: %.4f seconds\n', time_inference);
end

% 7. Identify the Best Model
[best_f1, best_idx] = max(f1_scores);
best_model_name = model_names{best_idx};
best_model = models{best_idx};
fprintf('\nBest Fused ML Model: %s with Macro F1-Score: %.4f\n', best_model_name, best_f1);

% Save the best model
best_model_path = fullfile(models_dir, 'best_morphology_model.mat');
save(best_model_path, 'best_model', 'mu', 'sigma', 'best_model_name', 'accuracies', 'f1_scores', 'model_names');
fprintf('Saved best morphology (fused) model to %s\n', best_model_path);

% Plot and save confusion matrix of the best model
figure('Visible', 'off');
cm = confusionchart(Y_test, predictions{best_idx});
cm.Title = sprintf('Confusion Matrix: %s (Fused Features)', best_model_name);
saveas(gcf, fullfile(reports_dir, 'confusion_matrix_morphology.png'));
close(gcf);
fprintf('Saved confusion matrix plot to reports/confusion_matrix_morphology.png\n');

% Save evaluation variables for comparison script
save(fullfile(reports_dir, 'morphology_results.mat'), 'Y_test', 'predictions', ...
    'model_names', 'accuracies', 'f1_scores', 'time_svm_train', 'time_rf_train', 'time_knn_train');

fprintf('Fused ML Training and Evaluation Completed!\n');
