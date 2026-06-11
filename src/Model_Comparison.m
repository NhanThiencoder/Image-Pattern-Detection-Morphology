%% Model_Comparison.m
% So sánh tổng hợp 4 kết quả: SVM (Fusion), RF (Fusion), SVM (BoVW), RF (BoVW)
clear; clc; close all;

script_dir = fileparts(mfilename('fullpath'));
reports_dir = fullfile(script_dir, '..', 'reports');

% Load kết quả
try
    load(fullfile(reports_dir, 'fusion_results.mat'));
    load(fullfile(reports_dir, 'bovw_results.mat'));
catch
    error('Chưa có đủ file kết quả. Hãy chạy train_fusion.m và train_bovw.m trước!');
end

% Tạo bảng tổng hợp 4 kịch bản
Models = {'SVM (Fusion 24D)'; 'Random Forest (Fusion 24D)'; 'SVM (BoVW 100D)'; 'Random Forest (BoVW 100D)'};
Accuracy = [accuracies_fusion(1); accuracies_fusion(2); accuracies_bovw(1); accuracies_bovw(2)];
Macro_F1 = [f1_scores_fusion(1); f1_scores_fusion(2); f1_scores_bovw(1); f1_scores_bovw(2)];
Training_Time = [time_svm_fusion; time_rf_fusion; time_svm_bovw; time_rf_bovw];

FinalComparison = table(Models, Accuracy, Macro_F1, Training_Time);
disp('=== BẢNG TỔNG HỢP SO SÁNH 4 KỊCH BẢN ===');
disp(FinalComparison);

% Vẽ biểu đồ cột gộp
figure('Name', 'So sánh 4 Kịch bản', 'Position', [100, 100, 800, 500]);
bar_data = [Accuracy, Macro_F1];
b = bar(bar_data);
set(gca, 'XTickLabel', Models, 'XTickLabelRotation', 15);
ylabel('Điểm số (0 - 1.0)');
legend({'Accuracy', 'Macro F1-Score'}, 'Location', 'northeast');
title('So sánh Hiệu năng: Phương pháp Trích xuất và Mô hình phân lớp');
grid on;

saveas(gcf, fullfile(reports_dir, 'final_4_models_comparison.png'));
disp('Đã lưu biểu đồ so sánh vào thư mục reports/');