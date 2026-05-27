%% RÚT TRÍCH ĐẶC TRƯNG GLCM VÀ MÀU SẮC (HSV) TỪ ẢNH ĐÃ TÁCH NỀN
warning('off', 'all');
script_dir = fileparts(mfilename('fullpath'));

segmented_dir = fullfile(script_dir, '..', 'data','segmented', 'dataset_segmented');

if ~exist(segmented_dir, 'dir')
    error(['Không tìm thấy dữ liệu tại: ', segmented_dir, ...
           newline, 'Hãy kiểm tra lại xem thư mục data có nằm cùng cấp với models không.']);
end

imds_seg = imageDatastore(segmented_dir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

num_images = numel(imds_seg.Files);
features_table = table(); 

h = waitbar(0, 'Đang trích xuất Texture & Color (Tốc độ cao)... Vui lòng đợi');

for i = 1:num_images
    img_path = imds_seg.Files{i};
    label = string(imds_seg.Labels(i)); 

    % Đọc ảnh đã tách nền
    img = imread(img_path);
    img = imresize(img, [256 256]);

    % --- KHÔI PHỤC MẶT NẠ (SIÊU NHANH) ---
    % Vì phông nền là màu đen (0,0,0), ta chỉ cần lấy pixel nào > 0 là ra ngay chiếc lá!
    mask = (img(:,:,1) > 0) | (img(:,:,2) > 0) | (img(:,:,3) > 0);

    if sum(mask(:)) == 0; continue; end % Bỏ qua nếu ảnh rỗng đen hoàn toàn

    % ====================================================================
    % PHẦN 1: MÀU SẮC (HSV)
    % ====================================================================
    hsv_img = rgb2hsv(img);
    H = hsv_img(:,:,1); S = hsv_img(:,:,2); V = hsv_img(:,:,3);

    H_leaf = H(mask); S_leaf = S(mask); V_leaf = V(mask);

    h_mean = mean(H_leaf); h_std = std(H_leaf); h_skew = skewness(H_leaf);
    s_mean = mean(S_leaf); s_std = std(S_leaf); s_skew = skewness(S_leaf);
    v_mean = mean(V_leaf); v_std = std(V_leaf); v_skew = skewness(V_leaf);

    % ====================================================================
    % PHẦN 2: KẾT CẤU (GLCM)
    % ====================================================================
    gray_img = im2gray(img);
    gray_double = double(gray_img);

    % Gán nền đen thành NaN để graycomatrix bỏ qua
    gray_double(~mask) = NaN;

    % GLCM 4 hướng
    offsets = [0 1; -1 1; -1 0; -1 -1]; 
    glcm = graycomatrix(gray_double, 'Offset', offsets, 'NumLevels', 256, 'Symmetric', true);
    stats = graycoprops(glcm, {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});

    glcm_contrast = mean(stats.Contrast);
    glcm_correlation = mean(stats.Correlation);
    glcm_energy = mean(stats.Energy);
    glcm_homogeneity = mean(stats.Homogeneity);

    % ====================================================================
    % LƯU BẢNG DỮ LIỆU
    % ====================================================================
    new_row = table(label, h_mean, h_std, h_skew, s_mean, s_std, s_skew, v_mean, v_std, v_skew, ...
        glcm_contrast, glcm_correlation, glcm_energy, glcm_homogeneity, ...
        'VariableNames', {'Label', ...
        'H_Mean', 'H_Std', 'H_Skew', ...
        'S_Mean', 'S_Std', 'S_Skew', ...
        'V_Mean', 'V_Std', 'V_Skew', ...
        'GLCM_Contrast', 'GLCM_Correlation', 'GLCM_Energy', 'GLCM_Homogeneity'});

    features_table = [features_table; new_row];

    if mod(i, 100) == 0
        waitbar(i / num_images, h, sprintf('Đang xử lý: %d / %d ảnh...', i, num_images));
    end
end
close(h);

% Xuất ra file CSV
writetable(features_table, 'dataset_texture_color.csv');
disp('✅ HOÀN TẤT!');
disp(['Đã lưu ', num2str(height(features_table)), ' dòng dữ liệu vào file: dataset_texture_color.csv']);