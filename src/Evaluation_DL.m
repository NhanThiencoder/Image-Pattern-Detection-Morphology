%% Deep Learning CNN Model Evaluation
% Đánh giá chi tiết mô hình Mạng nơ-ron Tích chập (CNN) tự học từ ảnh gốc.

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
    reports_dir = fullfile(pwd, 'reports');
else
    reports_dir = fullfile(project_root, 'reports');
end

results_path = fullfile(reports_dir, 'cnn_results.mat');

if ~exist(results_path, 'file')
    error('Chưa tìm thấy kết quả CNN. Vui lòng chạy train_cnn.m trước.');
end

% Load kết quả
load(results_path);

%% 2. Hiệu năng tổng quát của CNN
fprintf('Hiệu năng tổng quát của mô hình Custom CNN:\n');
fprintf('  - Validation Accuracy: %.4f (%.2f%%)\n', acc_cnn, acc_cnn * 100);
fprintf('  - Macro F1-Score: %.4f\n', macro_f1_cnn);
fprintf('  - Thời gian huấn luyện (CPU): %.2f giây\n', time_cnn_train);
fprintf('  - Thời gian suy diễn (Inference) trên 750 ảnh: %.4f giây\n', time_cnn_inference);

%% 3. Tính toán chỉ số chi tiết cho từng lớp bệnh
C = confusionmat(Y_val, YPred);
class_names = unique(string(Y_val));
num_classes = numel(class_names);

precisions = zeros(num_classes, 1);
recalls = zeros(num_classes, 1);
f1s = zeros(num_classes, 1);

for c = 1:num_classes
    tp = C(c, c);
    fp = sum(C(:, c)) - tp;
    fn = sum(C(c, :)) - tp;
    
    if (tp + fp) > 0, precisions(c) = tp / (tp + fp); else, precisions(c) = 0; end
    if (tp + fn) > 0, recalls(c) = tp / (tp + fn); else, recalls(c) = 0; end
    if (precisions(c) + recalls(c)) > 0
        f1s(c) = 2 * (precisions(c) * recalls(c)) / (precisions(c) + recalls(c));
    else
        f1s(c) = 0;
    end
end

% Hiển thị bảng chỉ số đo lường chi tiết
ClassMetricsTable = table(class_names, precisions, recalls, f1s, ...
    'VariableNames', {'Class_Name', 'Precision', 'Recall', 'F1_Score'});
disp('Bảng chi tiết chỉ số đo lường của từng lớp bệnh đối với Custom CNN:');
disp(ClassMetricsTable);

%% 4. Trực quan hóa Ma trận Nhầm lẫn (Confusion Matrix)
figure('Name', 'Confusion Matrix - Custom CNN');
cm = confusionchart(Y_val, YPred);
cm.Title = 'Ma trận Nhầm lẫn: Custom CNN (Học sâu từ ảnh gốc)';

%% 5. Phân tích & Đặc điểm của Học sâu CNN
%
% 1. **Khả năng tự học đặc trưng (Self-learning Features)**:
%    - Điểm mạnh lớn nhất của CNN là khả năng **tự động học** các bộ lọc đặc trưng từ ảnh RGB thô 
%      (qua các lớp Convolutional và Max Pooling) thay vì phải thiết kế thủ công như hình thái học.
%    - Nhờ vậy, CNN nhận diện tốt các kết cấu vết bệnh phức tạp, vùng lá hoại tử hoặc gân lá biến đổi màu.
%
% 2. **Hạn chế về tính giải thích (Black Box)**:
%    - CNN hoạt động như một "hộp đen". Các bộ lọc tích chập ở các tầng sâu đại diện cho các đặc trưng 
%      toán học phức tạp không thể trực quan hóa hay mô tả bằng từ ngữ sinh học.
%    - Chúng ta không thể biết chính xác mô hình phân loại lá bệnh dựa trên "diện tích đốm bệnh cụ thể" 
%      hay do các yếu tố nhiễu môi trường của ảnh nền.
%
% 3. **Chi phí tính toán cao (Computational Bottleneck)**:
%    - Chỉ huấn luyện trên một tập con nhỏ (750 ảnh) và chạy 8 epoch, mô hình đã mất tới **44.70 giây** 
%      trên CPU. Điều này cho thấy nếu tăng lên toàn bộ 20,000 ảnh, thời gian huấn luyện có thể kéo dài 
%      hàng giờ liền nếu không có sự hỗ trợ của GPU tăng tốc chuyên dụng.
