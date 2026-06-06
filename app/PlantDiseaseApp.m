function PlantDiseaseApp()
    % TẠO CỬA SỔ CHÍNH (MAIN FIGURE)
    fig = uifigure('Name', 'Hệ Thống Phân Tích Sâu Bệnh (Hybrid Model)', ...
                   'Position', [100, 100, 1100, 650]);
    
    % Biến toàn cục (State) để lưu trữ dữ liệu trong App
    appData = struct('OriginalImage', [], 'MarkedImage', [], 'DiseaseMask', []);

    % CHIA BỐ CỤC CHÍNH LÀM 2 CỘT (Grid Layout)
    % Cột 1: Bảng điều khiển (Rộng 350px) | Cột 2: Hiển thị ảnh (Tự động co giãn '1x')
    mainGrid = uigridlayout(fig, [1, 2]);
    mainGrid.ColumnWidth = {350, '1x'};

    % =====================================================================
    % KHU VỰC 1: FRAME BẢNG ĐIỀU KHIỂN & KẾT QUẢ (CỘT TRÁI)
    % =====================================================================
    leftPanel = uipanel(mainGrid, 'Title', '⚙️ Điều Khiển & Kết Quả Học Thuật', ...
                        'FontSize', 14, 'FontWeight', 'bold');
    leftGrid = uigridlayout(leftPanel, [7, 1]);
    leftGrid.RowHeight = {40, 40, 40, 50, 150, '1x', 40}; % Chiều cao các hàng

    % 1. Nút Tải Ảnh
    uploadBtn = uibutton(leftGrid, 'push', 'Text', '📁 TẢI ẢNH LÊN', ...
                         'FontSize', 14, 'FontWeight', 'bold', ...
                         'ButtonPushedFcn', @uploadImageCallback);
    
    % 2. Dropdown Chọn Mô Hình
    uilabel(leftGrid, 'Text', 'Chọn Mô Hình (Feature Extraction):', 'FontWeight', 'bold');
    modelDropdown = uidropdown(leftGrid, ...
        'Items', {'1. Morphological + RF (Cơ bản)', '2. Texture + Color + RF', '3. Hybrid CNN-SVM (Tối ưu)'}, ...
        'FontSize', 12);

    % 3. Nút Phân Tích
    analyzeBtn = uibutton(leftGrid, 'push', 'Text', '🔍 PHÂN TÍCH & TRÍCH XUẤT', ...
                          'FontSize', 14, 'FontWeight', 'bold', ...
                          'BackgroundColor', [0.2 0.6 0.3], 'FontColor', 'white', ...
                          'Enable', 'off', ... % Ban đầu tắt, có ảnh mới bật
                          'ButtonPushedFcn', @analyzeImageCallback);

    % 4. Gauge: Đồng hồ đo Mật Độ Bệnh
    gaugeLabel = uilabel(leftGrid, 'Text', 'Mật độ Sâu Bệnh (Density %):', ...
                         'FontWeight', 'bold', 'HorizontalAlignment', 'center');
    densityGauge = uigauge(leftGrid, 'semicircular', 'Limits', [0 100], ...
                           'Value', 0);
    
    % 5. Bảng hiển thị Đặc Trưng Hình Thái (Morphological Features)
    uilabel(leftGrid, 'Text', 'Bảng Đặc Trưng Hình Thái (Morphological):', 'FontWeight', 'bold');
    featureTable = uitable(leftGrid, 'ColumnName', {'Đặc Trưng', 'Giá Trị'}, ...
                           'RowName', [], 'Data', cell(0,2));

    % 6. Nhãn kết luận bệnh
    resultLabel = uilabel(leftGrid, 'Text', 'Trạng thái: Chờ tải ảnh...', ...
                          'FontSize', 16, 'FontColor', [0.8 0.2 0.2], ...
                          'FontWeight', 'bold', 'HorizontalAlignment', 'center');

    % =====================================================================
    % KHU VỰC 2: FRAME TRỰC QUAN HÓA ẢNH (CỘT PHẢI)
    % =====================================================================
    rightPanel = uipanel(mainGrid, 'Title', '🖼️ Trực Quan Hóa (Visualizations)', ...
                         'FontSize', 14, 'FontWeight', 'bold');
    rightGrid = uigridlayout(rightPanel, [1, 2]); % 1 hàng, 2 cột ảnh

    % Ảnh gốc
    axOriginal = uiaxes(rightGrid);
    title(axOriginal, 'Ảnh Đầu Vào (Original Image)');
    axOriginal.XColor = 'none'; axOriginal.YColor = 'none'; % Ẩn trục tọa độ

    % Ảnh đã xử lý (Marked Features)
    axMarked = uiaxes(rightGrid);
    title(axMarked, 'Định vị Vết Bệnh (Marked Features)');
    axMarked.XColor = 'none'; axMarked.YColor = 'none';

    % =====================================================================
    % CÁC HÀM XỬ LÝ SỰ KIỆN (BACKEND LOGIC)
    % =====================================================================

    % --- Logic 1: Khi bấm nút "Tải Ảnh Lên" ---
    function uploadImageCallback(~, ~)
        [filename, pathname] = uigetfile({'*.jpg;*.jpeg;*.png', 'Image Files'});
        if isequal(filename, 0)
            return; % Hủy chọn file
        end
        
        % Đọc ảnh và lưu vào State
        fullpath = fullfile(pathname, filename);
        appData.OriginalImage = imread(fullpath);
        
        % Hiển thị ảnh lên giao diện
        imshow(appData.OriginalImage, 'Parent', axOriginal);
        
        % Reset các thông số cũ
        cla(axMarked); % Xóa ảnh kết quả cũ
        densityGauge.Value = 0;
        featureTable.Data = cell(0,2);
        resultLabel.Text = 'Sẵn sàng phân tích...';
        resultLabel.FontColor = [0.2 0.2 0.8];
        
        % Bật nút Phân tích
        analyzeBtn.Enable = 'on';
    end

    % --- Logic 2: Khi bấm nút "Phân Tích & Trích Xuất" ---
    function analyzeImageCallback(~, ~)
        % Thay đổi trạng thái UI đang tính toán
        analyzeBtn.Text = '⏳ Đang xử lý...';
        drawnow; % Ép MATLAB cập nhật giao diện ngay lập tức
        
        try
            img = appData.OriginalImage;
            
            % -------------------------------------------------------------
            % PHẦN CORE ALGORITHM: THAY BẰNG MODEL AI/CV CỦA NHÓM BẠN Ở ĐÂY
            % Dưới đây là code Xử lý ảnh cơ bản (Mock) để Demo thuật toán:
            % -------------------------------------------------------------
            
            % B1. Chuyển sang không gian màu HSV để tách lá và vết bệnh
            hsvImg = rgb2hsv(img);
            sChannel = hsvImg(:,:,2);
            
            % B2. Tạo mask chiếc lá (Tách nền) - Giả lập
            leafMask = sChannel > 0.2; 
            
            % B3. Phân đoạn vùng bệnh (Dùng Otsu trên kênh Gray)
            grayImg = rgb2gray(img);
            diseaseMask = (grayImg < 100) & leafMask; % Giả lập vết thối có màu tối
            
            % B4. Tính toán Morphological Features (Dùng regionprops)
            statsLeaf = regionprops(leafMask, 'Area', 'Perimeter');
            statsDisease = regionprops(diseaseMask, 'Area');
            
            totalLeafArea = sum([statsLeaf.Area]);
            totalDiseaseArea = sum([statsDisease.Area]);
            
            % Nếu không có lá, tránh lỗi chia cho 0
            if totalLeafArea == 0; totalLeafArea = 1; end
            
            % B5. Tính Mật độ (Density)
            density = (totalDiseaseArea / totalLeafArea) * 100;
            
            % Lấy một số đặc trưng phụ họa cho bảng
            leafPerimeter = sum([statsLeaf.Perimeter]);
            solidityMock = rand() * 0.2 + 0.7; % Giả lập độ lồi
            
            % -------------------------------------------------------------
            % CẬP NHẬT KẾT QUẢ LÊN GIAO DIỆN (UI)
            % -------------------------------------------------------------
            
            % Cập nhật Đồng hồ
            densityGauge.Value = min(density, 100);
            
            % Cập nhật Bảng Tabular Data
            featureTable.Data = {
                'Diện Tích Lá (px)', round(totalLeafArea);
                'Diện Tích Bệnh (px)', round(totalDiseaseArea);
                'Chu Vi Lá (px)', round(leafPerimeter);
                'Độ Lồi (Solidity)', sprintf('%.2f', solidityMock);
                'MẬT ĐỘ BỆNH (%)', sprintf('%.2f %%', density)
            };
            
            % Đưa ra kết luận dựa vào mật độ
            if density < 2
                resultLabel.Text = 'Kết luận: LÁ KHỎE MẠNH 🌿';
                resultLabel.FontColor = [0.2 0.8 0.2];
            else
                resultLabel.Text = 'Kết luận: PHÁT HIỆN SÂU BỆNH ⚠️';
                resultLabel.FontColor = [0.8 0.2 0.2];
            end
            
            % Vẽ viền đỏ lên ảnh gốc (Marked Features)
            imshow(img, 'Parent', axMarked);
            hold(axMarked, 'on');
            [B, ~] = bwboundaries(diseaseMask, 'noholes');
            for k = 1:length(B)
                boundary = B{k};
                plot(axMarked, boundary(:,2), boundary(:,1), 'r', 'LineWidth', 2);
            end
            hold(axMarked, 'off');
            
        catch ME
            uialert(fig, ['Có lỗi xảy ra trong quá trình xử lý: ' ME.message], 'Lỗi Algorithm');
        end
        
        % Trả lại trạng thái nút
        analyzeBtn.Text = '🔍 PHÂN TÍCH & TRÍCH XUẤT';
    end
end