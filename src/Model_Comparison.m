%% Model Evaluation & Comparison Notebook
% Đánh giá & So sánh giữa trích xuất đặc trưng hình thái (Morphological + ML) và học sâu (CNN).
%
% Ở notebook này, chúng ta tiến hành đánh giá chi tiết và so sánh đối chiếu hai hướng tiếp cận:
% 1. **Hand-crafted Morphology + Traditional ML**: Dựa trên đặc trưng hình thái học tự trích xuất
%    (Diện tích, Chu vi, Độ tròn, Độ lồi,...) rồi phân lớp qua các mô hình SVM, Random Forest, KNN.
% 2. **End-to-End Deep Learning (CNN)**: Cho phép mô hình tự học đặc trưng trực tiếp từ ảnh lá.

%% 1. Thiết lập Đường dẫn & Tải kết quả
clear; clc; close all;

% Định vị đường dẫn chạy công cụ bằng cách tìm thư mục gốc chứa 'reports' và 'models'
current_search_dir = pwd;
try
    mfile_path = fileparts(mfilename('fullpath'));
    if ~isempty(mfile_path)
        current_search_dir = mfile_path;
    end
catch
end

project_root = '';
for level = 1:5
    if exist(fullfile(current_search_dir, 'reports'), 'dir') && exist(fullfile(current_search_dir, 'models'), 'dir')
        project_root = current_search_dir;
        break;
    end
    parent_dir = fileparts(current_search_dir);
    if strcmp(parent_dir, current_search_dir)
        break;
    end
    current_search_dir = parent_dir;
end

if isempty(project_root)
    project_root = pwd;
end

reports_dir = fullfile(project_root, 'reports');
src_dir = fullfile(project_root, 'src');

morph_results_path = fullfile(reports_dir, 'morphology_results.mat');
cnn_results_path = fullfile(reports_dir, 'cnn_results.mat');

% Nếu chưa chạy huấn luyện, tự động gọi các file .m để huấn luyện trước
if ~exist(morph_results_path, 'file')
    fprintf('Chưa tìm thấy kết quả Morphology ML. Đang huấn luyện...\n');
    run(fullfile(src_dir, 'train_morphology_ml.m'));
end

if ~exist(cnn_results_path, 'file')
    fprintf('Chưa tìm thấy kết quả CNN. Đang huấn luyện...\n');
    run(fullfile(src_dir, 'train_cnn.m'));
end

% Load kết quả
morph_res = load(morph_results_path);
cnn_res = load(cnn_results_path);

%% 2. Tổng hợp Chỉ số Đo lường (Accuracy & Macro F1-Score)
% Trích xuất chỉ số của các mô hình truyền thống
model_names = morph_res.model_names;
accuracies = morph_res.accuracies;
f1_scores = morph_res.f1_scores;

% Thêm kết quả của CNN
model_names{end+1} = 'Custom CNN (Deep Learning)';
accuracies(end+1) = cnn_res.acc_cnn;
f1_scores(end+1) = cnn_res.macro_f1_cnn;

% Hiển thị bảng so sánh hiệu năng phân lớp
PerformanceTable = table(model_names', accuracies', f1_scores', ...
    'VariableNames', {'Model', 'Accuracy', 'Macro_F1_Score'});
disp('Bảng So Sánh Hiệu Năng Phân Lớp:');
disp(PerformanceTable);

%% 3. So sánh Thời gian Huấn luyện (Computational Time)
train_times = [morph_res.time_svm_train, ...
               morph_res.time_rf_train, ...
               morph_res.time_knn_train, ...
               cnn_res.time_cnn_train];

TimeTable = table(model_names', train_times', ...
    'VariableNames', {'Model', 'Training_Time_Seconds'});
disp('Bảng So Sánh Thời Gian Huấn Luyện (trên CPU):');
disp(TimeTable);

%% 4. Trực quan hóa So sánh Hiệu năng (Bar Chart)
figure('Name', 'Model Performance Comparison');
b = bar([accuracies; f1_scores]');
set(gca, 'XTickLabel', model_names, 'XTickLabelRotation', 15);
ylabel('Chỉ số (Score)');
legend({'Accuracy', 'Macro F1-Score'}, 'Location', 'northeastoutside');
title('So Sánh Hiệu Năng: Morphology ML vs. Deep Learning CNN');
grid on;

% Lưu đồ thị so sánh vào thư mục reports
saveas(gcf, fullfile(reports_dir, 'performance_comparison.png'));
fprintf('Đã lưu biểu đồ so sánh vào reports/performance_comparison.png\n');

%% 5. Biểu đồ Ma trận Nhầm lẫn (Confusion Matrix) của 2 mô hình tiêu biểu
% Chúng ta sẽ chọn mô hình Morphology ML tốt nhất và so sánh trực tiếp với CNN.
[best_morph_f1, best_morph_idx] = max(morph_res.f1_scores);
best_morph_name = morph_res.model_names{best_morph_idx};

% Hiển thị Confusion Matrix cho mô hình hình thái học tốt nhất
figure('Name', 'Confusion Matrix - Best Morphology ML');
cm_ml = confusionchart(morph_res.Y_test, morph_res.predictions{best_morph_idx});
cm_ml.Title = sprintf('Ma trận Nhầm lẫn: %s (Đặc trưng Hình thái)', best_morph_name);

% Hiển thị Confusion Matrix cho CNN học sâu
figure('Name', 'Confusion Matrix - Custom CNN');
cm_cnn = confusionchart(cnn_res.Y_val, cnn_res.YPred);
cm_cnn.Title = 'Ma trận Nhầm lẫn: Custom CNN (Học sâu từ ảnh gốc)';

%% 6. Phân Tích & Đánh Giá Chi Tiết (Key Insights)
%
% Hướng tiếp cận tự trích xuất đặc trưng (Hand-crafted Morphology) kết hợp với 
% mô hình học máy truyền thống và học sâu CNN có những ưu/nhược điểm rõ rệt 
% về mặt chi phí tính toán, độ chính xác và khả năng giải thích (explainability):
%
% 1. **Chi phí tính toán (Computational Complexity)**:
%    - *Morphology + ML*: Chi phí tính toán cực kỳ thấp. Quá trình huấn luyện các mô hình 
%      SVM, Random Forest, KNN chỉ mất khoảng **vài giây** trên CPU cho toàn bộ tập dữ liệu 
%      hơn 20,000 ảnh. Việc trích xuất đặc trưng hình thái (Diện tích lá, chu vi, tỉ lệ trục, độ tròn...) 
%      được xử lý nhanh chóng qua các phép toán hình học và morphology cơ bản.
%    - *Deep Learning (CNN)*: Chi phí tính toán rất cao. Ngay cả khi chỉ train trên một tập con 
%      nhỏ (750 ảnh) và chạy trong vài epoch ngắn ngủi, CNN vẫn mất **nhiều phút** trên CPU. 
%      Mạng CNN cần tối ưu hóa hàng triệu tham số thông qua lan truyền ngược (backpropagation) 
%      và các phép nhân ma trận dày đặc.
%
% 2. **Khả năng giải thích và tường minh (Interpretability/Explainability)**:
%    - *Morphology + ML*: **Rất cao (Hộp trắng)**. Chúng ta biết chính xác mô hình dựa vào các 
%      thuộc tính vật lý cụ thể để ra quyết định. Ví dụ, một chiếc lá nhiễm bệnh đốm lá (Spot) 
%      sẽ làm giảm độ lồi (Solidity) và tăng chu vi bất thường (Perimeter) do vết bệnh làm biến dạng 
%      hoặc tạo lỗ hổng. Cây quyết định (Decision Tree/Random Forest) có thể chỉ ra rõ ràng: 
%      "Nếu Diện tích vết bệnh > X pixel và Độ tròn < Y, thì lá bị nhiễm bệnh Z". Điều này có ý nghĩa 
%      sinh học thực tiễn cao, giúp các nhà nông nghiệp học tin tưởng và kiểm chứng quyết định của mô hình.
%    - *Deep Learning (CNN)*: **Thấp (Hộp đen)**. CNN tự học các bộ lọc đặc trưng (filters) phức tạp 
%      thông qua nhiều lớp tích chập. Biểu diễn của các lớp này là các vector không gian nhiều chiều 
%      không có ý nghĩa vật lý rõ ràng đối với con người. Một mô hình CNN phân loại lá bị "Blight" 
%      nhưng chúng ta không thể giải thích cụ thể nó dựa vào diện tích đốm bệnh hay hình dáng cấu trúc 
%      nào của lá, làm giảm tính an toàn khi triển khai thực tế.
%
% 3. **Độ chính xác và Khả năng tổng quát hóa**:
%    - CNN có ưu thế tự động học các pattern kết cấu (texture) phức tạp mà đặc trưng hình thái đơn giản 
%      không mô tả hết được (như sự thay đổi màu sắc tế bào, kết cấu gân lá...).
%    - Tuy nhiên, trong môi trường thực địa có nhiều nhiễu nền hoặc ánh sáng thay đổi, CNN thuần túy 
%      dễ bị quá khớp (overfitting) vào các chi tiết nền của phòng thí nghiệm (lab artifacts) thay vì 
%      học vết bệnh thực sự. Do đó, việc nắm giữ các đặc trưng hình thái học ổn định chính là "neo" 
%      vững chắc để mô hình học máy truyền thống đưa ra các quyết định có tính tổng quát tốt hơn.
