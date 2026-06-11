function PlantDiseaseApp()
    % TẠO CỬA SỔ CHÍNH (MAIN FIGURE)
    fig = uifigure('Name', 'Hệ Thống Phân Tích Sâu Bệnh (Hybrid Model)', ...
                   'Position', [100, 100, 1200, 750]);
    
    appData = struct('OriginalImage', [], 'MarkedImage', [], 'DiseaseMask', []);
    
    mainGrid = uigridlayout(fig, [1, 2]);
    mainGrid.ColumnWidth = {400, '1x'};
    
    % =====================================================================
    % KHU VỰC 1: FRAME BẢNG ĐIỀU KHIỂN & KẾT QUẢ
    % =====================================================================
    leftPanel = uipanel(mainGrid, 'Title', '⚙️ Điều Khiển & Kết Quả Học Thuật', ...
                        'FontSize', 14, 'FontWeight', 'bold');
                        
    % SỬA LỖI UI: Định nghĩa đúng 9 hàng cho 9 phần tử
    leftGrid = uigridlayout(leftPanel, [9, 1]);
    leftGrid.RowHeight = {40, 25, 35, 45, 25, 120, 25, '1x', 35}; 
    
    uploadBtn = uibutton(leftGrid, 'push', 'Text', '📁 TẢI ẢNH LÊN', ...
                         'FontSize', 14, 'FontWeight', 'bold', 'ButtonPushedFcn', @uploadImageCallback);
    
    uilabel(leftGrid, 'Text', 'Chọn Mô Hình AI (Classifier):', 'FontWeight', 'bold');
    modelDropdown = uidropdown(leftGrid, 'Items', {'Random Forest (Khuyên dùng)', 'SVM (RBF Kernel)'}, 'FontSize', 12);
        
    analyzeBtn = uibutton(leftGrid, 'push', 'Text', '🔍 PHÂN TÍCH & TRÍCH XUẤT', ...
                          'FontSize', 14, 'FontWeight', 'bold', ...
                          'BackgroundColor', [0.2 0.6 0.3], 'FontColor', 'white', ...
                          'Enable', 'off', 'ButtonPushedFcn', @analyzeImageCallback);
                          
    gaugeLabel = uilabel(leftGrid, 'Text', 'Mật độ Sâu Bệnh (Density %):', 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
    densityGauge = uigauge(leftGrid, 'semicircular', 'Limits', [0 100], 'Value', 0);
    
    uilabel(leftGrid, 'Text', 'Bảng Đặc Trưng Rút Trích (24 Features):', 'FontWeight', 'bold');
    featureTable = uitable(leftGrid, 'ColumnName', {'Đặc Trưng', 'Giá Trị'}, 'RowName', [], 'Data', cell(0,2));
                           
    resultLabel = uilabel(leftGrid, 'Text', 'Trạng thái: Chờ tải ảnh...', ...
                          'FontSize', 16, 'FontColor', [0.8 0.2 0.2], 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
                          
    % =====================================================================
    % KHU VỰC 2: FRAME TRỰC QUAN HÓA ẢNH 
    % =====================================================================
    rightPanel = uipanel(mainGrid, 'Title', '🖼️ Trực Quan Hóa (Visualizations)', ...
                         'FontSize', 14, 'FontWeight', 'bold');
    rightGrid = uigridlayout(rightPanel, [1, 2]);
    
    axOriginal = uiaxes(rightGrid); title(axOriginal, 'Ảnh Đầu Vào');
    axOriginal.XColor = 'none'; axOriginal.YColor = 'none'; 
    
    axMarked = uiaxes(rightGrid); title(axMarked, 'Định vị Vết Bệnh (Marked Features)');
    axMarked.XColor = 'none'; axMarked.YColor = 'none';
    
    % =====================================================================
    % HÀM XỬ LÝ 1: TẢI ẢNH LÊN
    % =====================================================================
    function uploadImageCallback(~, ~)
        [filename, pathname] = uigetfile({'*.jpg;*.jpeg;*.png', 'Image Files'});
        if isequal(filename, 0); return; end
        
        fullpath = fullfile(pathname, filename);
        img = imread(fullpath);
        appData.OriginalImage = imresize(img, [256 256]);
        
        imshow(appData.OriginalImage, 'Parent', axOriginal);
        cla(axMarked);
        densityGauge.Value = 0;
        featureTable.Data = cell(0,2);
        resultLabel.Text = 'Sẵn sàng phân tích...';
        resultLabel.FontColor = [0.2 0.2 0.8];
        analyzeBtn.Enable = 'on';
    end
    
    % =====================================================================
    % HÀM XỬ LÝ 2: CHẠY AI & TRÍCH XUẤT
    % =====================================================================
    function analyzeImageCallback(~, ~)
        analyzeBtn.Text = '⏳ Đang load Model & Xử lý...';
        analyzeBtn.Enable = 'off';
        drawnow;
        
        try
            app_dir = fileparts(mfilename('fullpath'));
            model_path = fullfile(app_dir, '..', 'models', 'best_fusion_models.mat');
            if ~exist(model_path, 'file')
                model_path = fullfile(app_dir, 'models', 'best_fusion_models.mat');
            end
            if ~exist(model_path, 'file')
                error(['Không tìm thấy model tại: ', model_path]);
            end
            load(model_path, 'rf_fusion', 'svm_fusion', 'mu_fusion', 'sigma_fusion');
            
            % -------------------------------------------------------------
            % TIỀN XỬ LÝ & TẠO MASK (DÙNG ẢNH MỜ ĐỂ KHÁNG NHIỄU)
            % -------------------------------------------------------------
            img = appData.OriginalImage; % Ảnh sắc nét nguyên bản
            img_smooth = imgaussfilt(img, 1.5); % Ảnh làm mờ để tách nền
            
            % BƯỚC 1: Tìm chiếc lá dựa trên ảnh mờ
            grayImg_smooth = rgb2gray(img_smooth);
            level = graythresh(grayImg_smooth);
            leafMask = imbinarize(grayImg_smooth, level);
            leafMask = imfill(leafMask, 'holes');
            leafMask = bwareaopen(leafMask, 500); 
            leafMask = bwareafilt(leafMask, 1); 
            
            % BƯỚC 2: Khoanh vùng bệnh dựa trên ảnh mờ
            hsvImg_smooth = rgb2hsv(img_smooth);
            H_smooth = hsvImg_smooth(:,:,1); 
            diseaseMask = (H_smooth < 0.15 | H_smooth > 0.45) & leafMask;
            diseaseMask = bwareaopen(diseaseMask, 40); 
            
            % -------------------------------------------------------------
            % TÍNH TOÁN ĐẶC TRƯNG HÌNH THÁI (Dựa trên Mask, không ảnh hưởng bởi màu)
            % -------------------------------------------------------------
            props = regionprops(leafMask, 'Area', 'Perimeter', 'Eccentricity', ...
                'Solidity', 'Extent', 'MajorAxisLength', 'MinorAxisLength', ...
                'EquivDiameter', 'ConvexArea');
                
            if isempty(props)
                error('Không nhận diện được chiếc lá trong ảnh!');
            end
            
            Area = props.Area; Perimeter = props.Perimeter; Eccentricity = props.Eccentricity;
            Solidity = props.Solidity; Extent = props.Extent; MajorAxis = props.MajorAxisLength;
            MinorAxis = props.MinorAxisLength; EquivDiameter = props.EquivDiameter;
            ConvexArea = props.ConvexArea; AspectRatio = MajorAxis / max(MinorAxis, eps);
            Circularity = (4 * pi * Area) / max(Perimeter^2, eps);
            
            morph_features = [Area, Perimeter, Eccentricity, Solidity, Extent, ...
                              MajorAxis, MinorAxis, AspectRatio, EquivDiameter, ConvexArea, Circularity];
                              
            % -------------------------------------------------------------
            % TÍNH TOÁN MÀU SẮC & KẾT CẤU (BẮT BUỘC DÙNG ẢNH GỐC ĐỂ KHỚP VỚI LÚC TRAIN)
            % -------------------------------------------------------------
            % Quay lại dùng 'img' thay vì 'img_smooth'
            hsvImg_original = rgb2hsv(img); 
            H = hsvImg_original(:,:,1); S = hsvImg_original(:,:,2); V = hsvImg_original(:,:,3);
            
            H_leaf = H(leafMask); S_leaf = S(leafMask); V_leaf = V(leafMask);
            h_mean = mean(H_leaf); h_std = std(H_leaf); h_skew = skewness(H_leaf);
            s_mean = mean(S_leaf); s_std = std(S_leaf); s_skew = skewness(S_leaf);
            v_mean = mean(V_leaf); v_std = std(V_leaf); v_skew = skewness(V_leaf);
            
            gray_original = im2gray(img);
            gray_double = double(gray_original);
            gray_double(~leafMask) = NaN; % Bỏ nền đen bằng NaN
            
            offsets = [0 1; -1 1; -1 0; -1 -1]; 
            warning('off', 'images:graycomatrix:ignoreNaN');
            glcm = graycomatrix(gray_double, 'Offset', offsets, 'NumLevels', 256, 'Symmetric', true);
            stats = graycoprops(glcm, {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});
            warning('on', 'images:graycomatrix:ignoreNaN');
            
            color_texture_features = [h_mean, h_std, h_skew, s_mean, s_std, s_skew, ...
                                      v_mean, v_std, v_skew, mean(stats.Contrast), ...
                                      mean(stats.Correlation), mean(stats.Energy), mean(stats.Homogeneity)];
                                      
            % -------------------------------------------------------------
            % DỰ ĐOÁN (INFERENCE)
            % -------------------------------------------------------------
            X_input = [morph_features, color_texture_features];
            X_input(isnan(X_input)) = 0; 
            
            % Sửa biến mu, sigma thành mu_fusion, sigma_fusion
            X_input_scaled = (X_input - mu_fusion) ./ sigma_fusion; 
            
            % Sửa biến model thành rf_fusion và svm_fusion
            if contains(modelDropdown.Value, 'Random Forest')
                predicted_label = predict(rf_fusion, X_input_scaled);
            else
                predicted_label = predict(svm_fusion, X_input_scaled);
            end
            
            diseaseName = string(predicted_label);
            
            % -------------------------------------------------------------
            % CẬP NHẬT GIAO DIỆN
            % -------------------------------------------------------------
           statsDisease = regionprops(diseaseMask, 'Area');
            totalDiseaseArea = sum([statsDisease.Area]);
            density = (totalDiseaseArea / Area) * 100;
            
            % Phân rã logic kiểm tra
            ai_predicted_healthy = contains(lower(diseaseName), 'healthy');
            
            % XỬ LÝ XUNG ĐỘT (HYBRID FUSION LOGIC)
            if ai_predicted_healthy
                % Trường hợp 1: AI đoán đúng lá khỏe
                density = 0; 
                diseaseMask = false(size(diseaseMask)); 
                resultLabel.Text = ['🌿 LÁ KHỎE MẠNH (AI: ', char(diseaseName), ')'];
                resultLabel.FontColor = [0.1 0.6 0.1];
                
            elseif density < 1.0
                % Trường hợp 2: AI đoán sai (có bệnh), nhưng Hình thái học phủ quyết (không thấy vết bệnh)
                density = 0; 
                diseaseMask = false(size(diseaseMask)); 
                resultLabel.Text = '🌿 LÁ KHỎE MẠNH';
                resultLabel.FontColor = [0.1 0.6 0.1];
                
            else
                % Trường hợp 3: AI đoán có bệnh, và Hình thái học xác nhận có vết bệnh (> 1%)
                resultLabel.Text = ['⚠️ BỆNH LÝ: ', char(diseaseName)];
                resultLabel.FontColor = [0.8 0.1 0.1];
            end
            
            densityGauge.Value = min(density, 100);
            
            featureTable.Data = {
                '1. Morph: Diện Tích (px)', round(Area);
                '2. Morph: Chu Vi (px)', round(Perimeter);
                '3. Morph: Độ Dẹt', sprintf('%.3f', Eccentricity);
                '4. Morph: Độ Lồi', sprintf('%.3f', Solidity);
                '5. Morph: Độ Bao Phủ', sprintf('%.3f', Extent);
                '6. Morph: Trục Lớn', sprintf('%.2f', MajorAxis);
                '7. Morph: Trục Nhỏ', sprintf('%.2f', MinorAxis);
                '8. Morph: Tỷ Lệ Khung', sprintf('%.3f', AspectRatio);
                '9. Morph: Đ.Kính T.Đương', sprintf('%.2f', EquivDiameter);
                '10. Morph: Diện Tích Lồi', round(ConvexArea);
                '11. Morph: Độ Tròn', sprintf('%.3f', Circularity);
                '12. UI: MẬT ĐỘ BỆNH (%)', sprintf('%.2f %%', density);
                '13. Color: H-Mean (Màu sắc)', sprintf('%.3f', h_mean);
                '14. Color: H-Std', sprintf('%.3f', h_std);
                '15. Color: H-Skew', sprintf('%.3f', h_skew);
                '16. Color: S-Mean (Bão hòa)', sprintf('%.3f', s_mean);
                '17. Color: S-Std', sprintf('%.3f', s_std);
                '18. Color: S-Skew', sprintf('%.3f', s_skew);
                '19. Color: V-Mean (Độ sáng)', sprintf('%.3f', v_mean);
                '20. Color: V-Std', sprintf('%.3f', v_std);
                '21. Color: V-Skew', sprintf('%.3f', v_skew);
                '22. GLCM: Contrast', sprintf('%.3f', mean(stats.Contrast));
                '23. GLCM: Correlation', sprintf('%.3f', mean(stats.Correlation));
                '24. GLCM: Energy', sprintf('%.3f', mean(stats.Energy));
                '25. GLCM: Homogeneity', sprintf('%.3f', mean(stats.Homogeneity))
            };
            
           
            
            imshow(img, 'Parent', axMarked);
            hold(axMarked, 'on');
            [B, ~] = bwboundaries(diseaseMask, 'noholes');
            for k = 1:length(B)
                boundary = B{k};
                plot(axMarked, boundary(:,2), boundary(:,1), 'r', 'LineWidth', 2.5);
            end
            hold(axMarked, 'off');
            
        catch ME
            uialert(fig, ['Lỗi hệ thống: ' ME.message], 'Error');
        end
        
        analyzeBtn.Text = '🔍 PHÂN TÍCH & TRÍCH XUẤT';
        analyzeBtn.Enable = 'on';
    end
end