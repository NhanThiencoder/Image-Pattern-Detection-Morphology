%% train_bovw.m
% Huấn luyện mô hình SVM và Random Forest sử dụng đặc trưng BoVW (100D).
warning('off', 'all');
fprintf('=== TRAINING SVM & RANDOM FOREST ON BoVW FEATURES (100D) ===\n');

script_dir = fileparts(mfilename('fullpath'));
bovw_csv_path = fullfile(script_dir, '..', 'data', 'processed', 'dataset_keypoint_bovw.csv');
models_dir = fullfile(script_dir, '..', 'models');
reports_dir = fullfile(script_dir, '..', 'reports');

if ~exist(bovw_csv_path, 'file'), error('File BoVW CSV khong ton tai.'); end

% Đọc dữ liệu
data_bovw = readtable(bovw_csv_path);
X = data_bovw{:, 1:end-1};
Y = categorical(data_bovw.Label);

num_classes = numel(categories(Y));

% Chia tập Train/Test (80/20) - Giữ cùng random seed 42 để công bằng
rng(42);
cv = cvpartition(Y, 'Holdout', 0.2);
X_train = X(training(cv), :); Y_train = Y(training(cv));
X_test = X(test(cv), :);      Y_test = Y(test(cv));

% Chuẩn hóa Z-Score
mu_bovw = mean(X_train); sigma_bovw = std(X_train); sigma_bovw(sigma_bovw == 0) = 1;
X_train_scaled = (X_train - mu_bovw) ./ sigma_bovw;
X_test_scaled = (X_test - mu_bovw) ./ sigma_bovw;

% Train SVM
fprintf('Đang huấn luyện SVM (RBF) trên BoVW...\n'); tic;
t_svm = templateSVM('KernelFunction', 'rbf', 'Standardize', false);
svm_bovw = fitcecoc(X_train_scaled, Y_train, 'Learners', t_svm);
time_svm_bovw = toc;

% Train RF
fprintf('Đang huấn luyện Random Forest trên BoVW...\n'); tic;
t_tree = templateTree('MaxNumSplits', 100);
rf_bovw = fitcensemble(X_train_scaled, Y_train, 'Method', 'Bag', 'NumLearningCycles', 100, 'Learners', t_tree);
time_rf_bovw = toc;

% Đánh giá mô hình
models = {svm_bovw, rf_bovw};
model_names = {'SVM (BoVW)', 'RF (BoVW)'};
f1_scores_bovw = zeros(1, 2); accuracies_bovw = zeros(1, 2); predictions_bovw = cell(1, 2);

for m = 1:2
    pred = predict(models{m}, X_test_scaled);
    predictions_bovw{m} = pred;
    C = confusionmat(Y_test, pred);
    accuracies_bovw(m) = sum(diag(C)) / sum(C(:));
    
    % Macro F1
    f1s = zeros(num_classes, 1);
    for c = 1:num_classes
        tp = C(c, c); fp = sum(C(:, c)) - tp; fn = sum(C(c, :)) - tp;
        pr = tp/(tp+fp); re = tp/(tp+fn);
        if isnan(pr), pr=0; end; if isnan(re), re=0; end;
        if (pr+re)>0, f1s(c) = 2*(pr*re)/(pr+re); end
    end
    f1_scores_bovw(m) = mean(f1s);
end

% Lưu kết quả
save(fullfile(models_dir, 'best_bovw_models.mat'), 'svm_bovw', 'rf_bovw', 'mu_bovw', 'sigma_bovw');
save(fullfile(reports_dir, 'bovw_results.mat'), 'Y_test', 'predictions_bovw', 'accuracies_bovw', 'f1_scores_bovw', 'time_svm_bovw', 'time_rf_bovw');
fprintf('Đã lưu kết quả BoVW.\n');