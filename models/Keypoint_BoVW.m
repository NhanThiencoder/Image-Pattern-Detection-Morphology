% =========================================================================
% SCRIPT TRÍCH XUẤT ĐẶC TRƯNG BoVW - SỬ DỤNG 100% DỮ LIỆU ĐÃ TÁCH NỀN
% Vị trí lưu file: src/Extract_BoVW.m
% =========================================================================

disp('Bắt đầu quá trình trích xuất đặc trưng BoVW...');

% -------------------------------------------------------------------------
% BƯỚC 1: ĐỌC DỮ LIỆU 
% -------------------------------------------------------------------------
datasetPath = fullfile('..', 'data', 'segmented', 'dataset_segmented'); 

if ~exist(datasetPath, 'dir')
    error('LỖI: Không tìm thấy thư mục ảnh. Hãy kiểm tra lại đường dẫn!');
end

imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

disp(['Đã load thành công danh sách ', num2str(numel(imds.Files)), ' bức ảnh.']);

% -------------------------------------------------------------------------
% BƯỚC 2: TẠO TỪ ĐIỂN HÌNH ẢNH VỚI 100% DỮ LIỆU
% -------------------------------------------------------------------------
disp('Đang chạy SURF và K-Means để tạo Từ điển (Số cụm = 100)...');

bag = bagOfFeatures(imds, ...
    'VocabularySize', 100, ...
    'PointSelection', 'Detector');

% -------------------------------------------------------------------------
% BƯỚC 3: MÃ HÓA ẢNH & VẼ BIỂU ĐỒ 
% -------------------------------------------------------------------------
disp('Đang mã hóa toàn bộ tập ảnh thành Vector đặc trưng (Histogram 1D)...');

% Bắt buộc phải chạy dòng encode này trước để sinh ra featureMatrix
featureMatrix = encode(bag, imds); 
labels = imds.Labels;

% Vẽ biểu đồ tần suất từ vựng
wordFrequencies = sum(featureMatrix, 1);
figure;
bar(wordFrequencies);
title('Biểu đồ phân bố Tần suất Từ vựng Hình ảnh (BoVW)');
xlabel('Cụm Từ vựng (1 đến 100)');
ylabel('Tổng số lần xuất hiện');

% -------------------------------------------------------------------------
% BƯỚC 4: LƯU KẾT QUẢ RA FILE CSV 
% -------------------------------------------------------------------------
disp('Đang định dạng dữ liệu để xuất ra CSV...');

numFeatures = size(featureMatrix, 2);
columnNames = cell(1, numFeatures);
for i = 1:numFeatures
    columnNames{i} = sprintf('F%d', i);
end

featureTable = array2table(featureMatrix, 'VariableNames', columnNames);
featureTable.Label = labels;

processedPath = fullfile('..', 'data', 'processed');
if ~exist(processedPath, 'dir')
    mkdir(processedPath);
end

savePath = fullfile(processedPath, 'dataset_keypoint_bovw.csv');
writetable(featureTable, savePath);

disp('=========================================================================');
disp(['TUYỆT VỜI! Đã hoàn thành. File CSV được lưu tại: ', savePath]);
disp('=========================================================================');