import cv2
import numpy as np

# ========================================================================
# 1. TIỀN XỬ LÝ ẢNH KỸ LƯỠNG (BẢO VỆ VIỀN & TRỊ BÓNG ĐỔ)
# ========================================================================
def pre_process_image(rgb_img):
    # Bước 1: Lọc song phương (Bilateral Filter)
    # Khử nhiễu phông nền nhưng KHÔNG làm nhòe viền mép lá
    blurred = cv2.bilateralFilter(rgb_img, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Bước 2: Chuyển sang LAB để xử lý ánh sáng
    lab = cv2.cvtColor(blurred, cv2.COLOR_RGB2Lab)
    l_channel, a, b = cv2.split(lab)
    
    # Bước 3: Áp dụng CLAHE lên kênh L (Ánh sáng)
    # Tự động nhận diện và cân bằng lại các vùng bóng đổ đậm
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    
    # Gộp lại thành không gian LAB đã được làm sạch ánh sáng
    enhanced_lab = cv2.merge((cl, a, b))
    
    return enhanced_lab

# ========================================================================
# 2. HỆ THỐNG TÁCH NỀN CHROMA (SẮC ĐỘ)
# ========================================================================
def compute_chroma(img_lab):
    a = img_lab[:, :, 1].astype(np.float32) - 128.0
    b = img_lab[:, :, 2].astype(np.float32) - 128.0
    chroma = np.sqrt(a**2 + b**2)
    return cv2.normalize(chroma, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

def fabryzzio_segmentation(img_path, img_size=(256, 256)):
    # 1. Đọc và chuẩn hóa kích thước ảnh
    bgr = cv2.imread(img_path)
    if bgr is None: return None, None, None, None
    rgb = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), img_size)
    
    # 2. GỌI BƯỚC TIỀN XỬ LÝ (Khử bóng, làm nét viền)
    enhanced_lab = pre_process_image(rgb)
    
    # 3. Tính độ rực màu (Chroma) từ ảnh đã được tiền xử lý
    chroma = compute_chroma(enhanced_lab)
    
    # Làm mờ nhẹ mặt nạ Chroma để tránh bị lỗ liti do gân lá
    chroma_blur = cv2.GaussianBlur(chroma, (5, 5), 0)
    
    # 4. Phân ngưỡng Otsu
    _, mask = cv2.threshold(chroma_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 5. Hình thái học (Làm sạch rác & Khép viền)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    
    # Dùng chổi (7x7) để đảm bảo không một mép lá nào bị mẻ
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    
    # 6. Tìm cụm lá to nhất và lấp lỗ
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(mask)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(final_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        
    # 7. Áp dụng mặt nạ cắt nền (Cắt trên ảnh Gốc RGB)
    result_rgb = cv2.bitwise_and(rgb, rgb, mask=final_mask)
    
    return rgb, final_mask, result_rgb, "Enhanced-Chroma"