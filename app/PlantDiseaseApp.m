function PlantDiseaseApp()
    % =====================================================================
    % TẠO CỬA SỔ CHÍNH (MAIN FIGURE)
    % =====================================================================
    fig = uifigure('Name', 'Hệ Thống Phân Tích Sâu Bệnh (Hybrid Model)', ...
                   'Position', [100, 100, 1300, 750]);
    
    % Sử dụng appData để lưu trữ các biến dùng chung giữa các hàm callback
    appData = struct('OriginalImage', [], 'axOriginal', [], 'axMask', [], 'axMarked', [], ...
                     'analyzeBtn', [], 'densityGauge', [], 'featureTable', [], 'resultLabel', []);
    
    mainGrid = uigridlayout(fig, [1, 2]);
    mainGrid.ColumnWidth = {400, '1x'};
    
    % =====================================================================
    % KHU VỰC 1: BẢNG ĐIỀU KHIỂN & KẾT QUẢ
    % =====================================================================
    leftPanel = uipanel(mainGrid, 'Title', '⚙️ Điều Khiển & Kết Quả Học Thuật', ...
                        'FontSize', 14, 'FontWeight', 'bold');
                        
    leftGrid = uigridlayout(leftPanel, [9, 1]);
    leftGrid.RowHeight = {40, 25, 35, 45, 25, 120, 25, '1x', 35}; 
    
    uibutton(leftGrid, 'push', 'Text', '📁 TẢI ẢNH LÊN', ...
             'FontSize', 14, 'FontWeight', 'bold', 'ButtonPushedFcn', @uploadImageCallback);
    
    uilabel(leftGrid, 'Text', 'Chọn Mô Hình AI (Classifier):', 'FontWeight', 'bold');
    modelDropdown = uidropdown(leftGrid, 'Items', {'Random Forest (Khuyên dùng)', 'SVM (RBF Kernel)'}, 'FontSize', 12);
        
    appData.analyzeBtn = uibutton(leftGrid, 'push', 'Text', '🔍 PHÂN TÍCH & TRÍCH XUẤT', ...
                          'FontSize', 14, 'FontWeight', 'bold', ...
                          'BackgroundColor', [0.2 0.6 0.3], 'FontColor', 'white', ...
                          'Enable', 'off', 'ButtonPushedFcn', @analyzeImageCallback);
                          
    uilabel(leftGrid, 'Text', 'Mật độ Sâu Bệnh (Density %):', 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
    appData.densityGauge = uigauge(leftGrid, 'semicircular', 'Limits', [0 100], 'Value', 0);
    
    uilabel(leftGrid, 'Text', 'Bảng Đặc Trưng Rút Trích (24 Features):', 'FontWeight', 'bold');
    appData.featureTable = uitable(leftGrid, 'ColumnName', {'Đặc Trưng', 'Giá Trị'}, 'RowName', [], 'Data', cell(0,2));
                           
    appData.resultLabel = uilabel(leftGrid, 'Text', 'Trạng thái: Chờ tải ảnh...', ...
                          'FontSize', 16, 'FontColor', [0.8 0.2 0.2], 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
                          
    % =====================================================================
    % KHU VỰC 2: FRAME TRỰC QUAN HÓA ẢNH (3 CỘT)
    % =====================================================================
    rightPanel = uipanel(mainGrid, 'Title', '🖼️ Trực Quan Hóa (Visualizations)', ...
                         'FontSize', 14, 'FontWeight', 'bold');
    rightGrid = uigridlayout(rightPanel, [1, 3]); 
    
    appData.axOriginal = uiaxes(rightGrid); title(appData.axOriginal, 'Ảnh Đầu Vào');
    appData.axOriginal.XColor = 'none'; appData.axOriginal.YColor = 'none'; 
    
    appData.axMask = uiaxes(rightGrid); title(appData.axMask, 'Mặt Nạ Vùng Bệnh (Mask)');
    appData.axMask.XColor = 'none'; appData.axMask.YColor = 'none';
    
    appData.axMarked = uiaxes(rightGrid); title(appData.axMarked, 'Định vị Vết Bệnh');
    appData.axMarked.XColor = 'none'; appData.axMarked.YColor = 'none';
    
    % =====================================================================
    % HÀM XỬ LÝ 1: TẢI ẢNH LÊN
    % =====================================================================
    function uploadImageCallback(~, ~)
        [filename, pathname] = uigetfile({'*.jpg;*.jpeg;*.png', 'Image Files'});
        if isequal(filename, 0); return; end
        
        fullpath = fullfile(pathname, filename);
        img = imread(fullpath);
        appData.OriginalImage = imresize(img, [256 256]);
        
        imshow(appData.OriginalImage, 'Parent', appData.axOriginal);
        cla(appData.axMask);
        cla(appData.axMarked);
        
        appData.densityGauge.Value = 0;
        appData.featureTable.Data = cell(0,2);
        appData.resultLabel.Text = 'Sẵn sàng phân tích...';
        appData.resultLabel.FontColor = [0.2 0.2 0.8];
        appData.analyzeBtn.Enable = 'on';
    end
    
    % =====================================================================
    % HÀM XỬ LÝ 2: CHẠY AI & TRÍCH XUẤT (HOÀN CHỈNH LOGIC)
    % =====================================================================
    function analyzeImageCallback(~, ~)
        appData.analyzeBtn.Text = '⏳ Đang load Model & Xử lý...';
        appData.analyzeBtn.Enable = 'off';
        drawnow;
        
        try
            % -------------------------------------------------------------
            % 1. LOAD MODEL ĐỘNG 
            % -------------------------------------------------------------
            app_dir = fileparts(mfilename('fullpath'));
            projectRoot = fileparts(app_dir); 
            
            is_rf = contains(modelDropdown.Value, 'Random Forest');
            if is_rf
                model_name = 'rf_fusion_model.mat';
            else
                model_name = 'svm_fusion_model.mat';
            end
            
            model_path = fullfile(projectRoot, 'models', model_name);
            if ~exist(model_path, 'file')
                model_path = fullfile(app_dir, 'models', model_name);
            end
            
            if ~exist(model_path, 'file')
                error(['Không tìm thấy model tại: ', model_path, char(10), ...
                       'Vui lòng đảm bảo bạn đã chạy file train_fusion.m!']);
            end
            
            modelData = load(model_path);
            if is_rf
                classifier = modelData.rf_fusion;
            else
                classifier = modelData.svm_fusion;
            end
            mu_val = modelData.mu_fusion;
            sigma_val = modelData.sigma_fusion;
            
            % -------------------------------------------------------------
            % 2. TIỀN XỬ LÝ & TẠO MASK
            % -------------------------------------------------------------
            img = appData.OriginalImage; 
            img_smooth = imgaussfilt(img, 1.5); 
            
            grayImg_smooth = rgb2gray(img_smooth);
            level = graythresh(grayImg_smooth);
            leafMask = imbinarize(grayImg_smooth, level);
            leafMask = imfill(leafMask, 'holes');
            leafMask = bwareaopen(leafMask, 500); 
            leafMask = bwareafilt(leafMask, 1); 
            
            hsvImg_smooth = rgb2hsv(img_smooth);
            H_smooth = hsvImg_smooth(:,:,1); 
            diseaseMask = (H_smooth < 0.15 | H_smooth > 0.45) & leafMask;
            diseaseMask = bwareaopen(diseaseMask, 40); 
            
            % -------------------------------------------------------------
            % 3. TRÍCH XUẤT 24 ĐẶC TRƯNG (HÌNH THÁI, MÀU SẮC, GLCM)
            % -------------------------------------------------------------
            % a. Đặc trưng hình thái
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
                              
            % b. Đặc trưng màu sắc (HSV)
            hsvImg_original = rgb2hsv(img); 
            H = hsvImg_original(:,:,1); S = hsvImg_original(:,:,2); V = hsvImg_original(:,:,3);
            
            H_leaf = H(leafMask); S_leaf = S(leafMask); V_leaf = V(leafMask);
            h_mean = mean(H_leaf); h_std = std(H_leaf); h_skew = skewness(H_leaf);
            s_mean = mean(S_leaf); s_std = std(S_leaf); s_skew = skewness(S_leaf);
            v_mean = mean(V_leaf); v_std = std(V_leaf); v_skew = skewness(V_leaf);
            
            % c. Đặc trưng kết cấu (GLCM)
            gray_original = im2gray(img);
            gray_double = double(gray_original);
            gray_double(~leafMask) = NaN; 
            
            offsets = [0 1; -1 1; -1 0; -1 -1]; 
            warning('off', 'images:graycomatrix:ignoreNaN');
            glcm = graycomatrix(gray_double, 'Offset', offsets, 'NumLevels', 256, 'Symmetric', true);
            stats = graycoprops(glcm, {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});
            warning('on', 'images:graycomatrix:ignoreNaN');
            
            color_texture_features = [h_mean, h_std, h_skew, s_mean, s_std, s_skew, ...
                                      v_mean, v_std, v_skew, mean(stats.Contrast), ...
                                      mean(stats.Correlation), mean(stats.Energy), mean(stats.Homogeneity)];
                                      
            % -------------------------------------------------------------
            % 4. DỰ ĐOÁN (INFERENCE)
            % -------------------------------------------------------------
            X_input = [morph_features, color_texture_features];
            X_input(isnan(X_input)) = 0; 
            X_input_scaled = (X_input - mu_val) ./ sigma_val; 
            
            predicted_label = predict(classifier, X_input_scaled);
            diseaseName = string(predicted_label);
            
            % -------------------------------------------------------------
            % 5. CẬP NHẬT GIAO DIỆN & XỬ LÝ HYBRID LOGIC
            % -------------------------------------------------------------
            statsDisease = regionprops(diseaseMask, 'Area');
            totalDiseaseArea = sum([statsDisease.Area]);
            density = (totalDiseaseArea / Area) * 100;
            
            ai_predicted_healthy = contains(lower(diseaseName), 'healthy');
            
            if ai_predicted_healthy
                density = 0; 
                diseaseMask = false(size(diseaseMask)); 
                appData.resultLabel.Text = ['🌿 LÁ KHỎE MẠNH (AI: ', char(diseaseName), ')'];
                appData.resultLabel.FontColor = [0.1 0.6 0.1];
            elseif density < 1
                density = 0; 
                diseaseMask = false(size(diseaseMask)); 
                appData.resultLabel.Text = '🌿 LÁ KHỎE MẠNH (Phủ quyết AI do mật độ <1%)';
                appData.resultLabel.FontColor = [0.1 0.6 0.1];
            else
                appData.resultLabel.Text = ['⚠️ BỆNH LÝ: ', char(diseaseName)];
                appData.resultLabel.FontColor = [0.8 0.1 0.1];
            end
            
            appData.densityGauge.Value = min(density, 100);
            
            appData.featureTable.Data = {
                '1. Morph: Diện Tích', round(Area);
                '2. Morph: Chu Vi', round(Perimeter);
                '3. Morph: Độ Dẹt', sprintf('%.3f', Eccentricity);
                '4. Morph: Độ Lồi', sprintf('%.3f', Solidity);
                '5. Morph: Độ Bao Phủ', sprintf('%.3f', Extent);
                '6. Morph: Trục Lớn', sprintf('%.2f', MajorAxis);
                '7. Morph: Trục Nhỏ', sprintf('%.2f', MinorAxis);
                '8. Morph: Tỷ Lệ Khung', sprintf('%.3f', AspectRatio);
                '9. Morph: ĐK Tương Đương', sprintf('%.2f', EquivDiameter);
                '10. Morph: Diện Tích Lồi', round(ConvexArea);
                '11. Morph: Độ Tròn', sprintf('%.3f', Circularity);
                '12. UI: MẬT ĐỘ BỆNH', sprintf('%.2f %%', density);
                '13. Color: H-Mean', sprintf('%.3f', h_mean);
                '14. Color: H-Std', sprintf('%.3f', h_std);
                '15. Color: H-Skew', sprintf('%.3f', h_skew);
                '16. Color: S-Mean', sprintf('%.3f', s_mean);
                '17. Color: S-Std', sprintf('%.3f', s_std);
                '18. Color: S-Skew', sprintf('%.3f', s_skew);
                '19. Color: V-Mean', sprintf('%.3f', v_mean);
                '20. Color: V-Std', sprintf('%.3f', v_std);
                '21. Color: V-Skew', sprintf('%.3f', v_skew);
                '22. GLCM: Contrast', sprintf('%.3f', mean(stats.Contrast));
                '23. GLCM: Correlation', sprintf('%.3f', mean(stats.Correlation));
                '24. GLCM: Energy', sprintf('%.3f', mean(stats.Energy));
                '25. GLCM: Homogeneity', sprintf('%.3f', mean(stats.Homogeneity))
            };
            
            % HIỂN THỊ HÌNH ẢNH LÊN 3 AXES
            imshow(img, 'Parent', appData.axOriginal);
            imshow(diseaseMask, 'Parent', appData.axMask);
            
            imshow(img, 'Parent', appData.axMarked);
            hold(appData.axMarked, 'on');
            [B, ~] = bwboundaries(diseaseMask, 'noholes');
            for k = 1:length(B)
                plot(appData.axMarked, B{k}(:,2), B{k}(:,1), 'r', 'LineWidth', 2.5);
            end
            hold(appData.axMarked, 'off');
            
        catch ME
            uialert(fig, ['Lỗi hệ thống: ' ME.message], 'Error');
        end
        
        appData.analyzeBtn.Text = '🔍 PHÂN TÍCH & TRÍCH XUẤT';
        appData.analyzeBtn.Enable = 'on';
    end
end