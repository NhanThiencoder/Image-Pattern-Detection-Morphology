%% train_fusion.m
% Huấn luyện mô hình SVM và Random Forest sử dụng đặc trưng dung hợp (Fusion 24D).
warning('off', 'all');
fprintf('=== TRAINING SVM & RANDOM FOREST ON FUSION FEATURES (24D) ===\n');

script_dir = fileparts(mfilename('fullpath'));
morph_csv_path = fullfile(script_dir, '..', 'data', 'processed', 'dataset_morphology.csv');
texture_csv_path = fullfile(script_dir, '..', 'data', 'processed', 'dataset_texture_color.csv');
models_dir = fullfile(script_dir, '..', 'models');
reports_dir = fullfile(script_dir, '..', 'reports');

if ~exist(reports_dir, 'dir'), mkdir(reports_dir); end

% Đọc dữ liệu
data_morph = readtable(morph_csv_path);
data_texture = readtable(texture_csv_path);
X = [data_morph{:, 2:end}, data_texture{:, 2:end}];
X = fillmissing(X, 'constant', 0);
Y = categorical(data_morph.Label);

num_classes = numel(categories(Y));

% Chia tập Train/Test (80/20)
rng(42);
cv = cvpartition(Y, 'Holdout', 0.2);
X_train = X(training(cv), :); Y_train = Y(training(cv));
X_test = X(test(cv), :);      Y_test = Y(test(cv));

% Chuẩn hóa Z-Score
mu_fusion = mean(X_train); sigma_fusion = std(X_train); sigma_fusion(sigma_fusion == 0) = 1;
X_train_scaled = (X_train - mu_fusion) ./ sigma_fusion;
X_test_scaled = (X_test - mu_fusion) ./ sigma_fusion;

% Train SVM
fprintf('Đang huấn luyện SVM (RBF)...\n'); tic;
t_svm = templateSVM('KernelFunction', 'rbf', 'Standardize', false);
svm_fusion = fitcecoc(X_train_scaled, Y_train, 'Learners', t_svm);
time_svm_fusion = toc;

% Train RF
fprintf('Đang huấn luyện Random Forest...\n'); tic;
t_tree = templateTree('MaxNumSplits', 100);
rf_fusion = fitcensemble(X_train_scaled, Y_train, 'Method', 'Bag', 'NumLearningCycles', 100, 'Learners', t_tree);
time_rf_fusion = toc;

% Đánh giá mô hình
models = {svm_fusion, rf_fusion};
model_names = {'SVM (Fusion)', 'RF (Fusion)'};
f1_scores_fusion = zeros(1, 2); accuracies_fusion = zeros(1, 2); predictions_fusion = cell(1, 2);

for m = 1:2
    pred = predict(models{m}, X_test_scaled);
    predictions_fusion{m} = pred;
    C = confusionmat(Y_test, pred);
    accuracies_fusion(m) = sum(diag(C)) / sum(C(:));
    
    % Macro F1
    f1s = zeros(num_classes, 1);
    for c = 1:num_classes
        tp = C(c, c); fp = sum(C(:, c)) - tp; fn = sum(C(c, :)) - tp;
        pr = tp/(tp+fp); re = tp/(tp+fn);
        if isnan(pr), pr=0; end; if isnan(re), re=0; end;
        if (pr+re)>0, f1s(c) = 2*(pr*re)/(pr+re); end
    end
    f1_scores_fusion(m) = mean(f1s);
end

% Lưu kết quả
save(fullfile(models_dir, 'best_fusion_models.mat'), 'svm_fusion', 'rf_fusion', 'mu_fusion', 'sigma_fusion');
save(fullfile(reports_dir, 'fusion_results.mat'), 'Y_test', 'predictions_fusion', 'accuracies_fusion', 'f1_scores_fusion', 'time_svm_fusion', 'time_rf_fusion');
fprintf('Đã lưu kết quả Fusion.\n');