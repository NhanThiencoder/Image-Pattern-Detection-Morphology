# Đề tài: Xử lý Ảnh và Đặc trưng Hình thái (Morphological Features) trong Phát hiện và Phân tích Mật độ Bệnh trên Lá cây.
## Ứng dụng Morphological Hit-or-Miss và Marked Features (Morphological Reconstruction)

## 👥 Thành viên
- Hồ Đức Nhân Thiện - 31231026999
- Trần Quốc Bảo - 31231022657
- Nguyễn Huỳnh Tấn Phát - 31231024397
- Nguyễn Ngọc Minh Đức - 31231020325

---

## Giới thiệu (Introduction)
Dự án này là Đồ án cuối kỳ môn Xử lý và Phân tích Hình ảnh. Mục tiêu của dự án là nghiên cứu, cài đặt và đánh giá hiệu quả của các phương pháp Hình thái học (Morphological) và Đặc trưng đánh dấu (Marked Features) để phát hiện và trích xuất các mẫu (pattern) cụ thể trong ảnh kỹ thuật số.

### Bài toán ứng dụng cụ thể: Nhận diện và phân tích mật độ vùng nhiễm sâu bệnh trên lá cây (Nông nghiệp số).
- Hit-or-Miss Transform được sử dụng để dò tìm chính xác các đốm bệnh nhỏ hoặc các viền cấu trúc bất thường trên bề mặt lá dựa vào các structuring elements (SE) thiết kế sẵn.
- Morphological Reconstruction (Marked Features) được áp dụng để khôi phục và giữ nguyên hình dáng, kích thước của các mảng bệnh lớn từ các "hạt giống" (markers) ban đầu, giúp khoanh vùng chính xác vết bệnh mà không làm thay đổi đặc tính hình học, đồng thời loại bỏ nhiễu nền.

---

## Tính năng chính (Features)
- Tiền xử lý ảnh (Khử nhiễu, binarize, cân bằng sáng).
- Phát hiện pattern sử dụng Morphological Hit-or-Miss Transform.
- Trích xuất cấu trúc phức tạp sử dụng Marked Features và Morphological Reconstruction.
- Giao diện Web tương tác trực quan cho phép người dùng upload ảnh và tinh chỉnh tham số theo thời gian thực.
- Bảng module so sánh hiệu năng (Accuracy, Processing Time) giữa các thuật toán.

## Mục tiêu Đồ án
- Làm sạch và Phân đoạn ảnh
- Cài đặt thuật toán cốt lỗi
- Định lượng mật độ
- So sánh thuật toán
- Triển khai Web App
  
## Dữ liệu (Dataset)
Dự án sử dụng bộ dữ liệu chuẩn hóa PlantVillage từ Kaggle.
- Nguồn: Kaggle - Plant Disease Dataset
- Đặc điểm: Bao gồm hình ảnh màu (RGB) của các loại lá cây ở nhiều trạng thái: Khỏe mạnh (Healthy) và Nhiễm bệnh (Blight, Spot, Rust, Rot,...).
- Tiền xử lý: 
  
## Cấu trúc hệ thống 
---

## 📌 Workflow làm việc nhóm trên GitHub
### 1) Quy tắc Issue
- Mỗi task chính tạo **1 Issue** (gán assignee, label).
- Mẫu label gợi ý:
  - `dataset`, `annotation`, `baseline`, `morphology`, `web`, `evaluation`, `docs`

### 2) Quy tắc Branch
- Không commit trực tiếp lên nhánh mặc định.
- Tạo branch theo format:
  - `feature/<ten-tinh-nang>` (ví dụ: `feature/hit-or-miss`)
  - `fix/<ten-loi>`
  - `docs/<noi-dung>`

### 3) Quy tắc Pull Request (PR)
- Mỗi PR chỉ tập trung 1 nhóm thay đổi.
- PR description cần có:
  - Tóm tắt thay đổi
  - Cách test (lệnh chạy)
  - Screenshot kết quả (nếu có)
- Ít nhất **1 thành viên review** trước khi merge.

### 4) Commit message convention
- Gợi ý format:
  - `feat: ...` thêm tính năng
  - `fix: ...` sửa lỗi
  - `docs: ...` tài liệu
  - `refactor: ...` cải tiến code không đổi chức năng

### 5) Definition of Done (DoD)
Một task được xem là xong khi:
- Có PR merge vào nhánh mặc định
- Chạy được tối thiểu 1 test/demo (ví dụ chạy 10 ảnh mẫu)
- Có cập nhật docs (nếu liên quan)

---

## Đánh giá & So sánh (gợi ý)
- Metrics: Precision, Recall, F1-score (theo khoảng cách tâm), Processing Time.
- So sánh tối thiểu:
  - Morphology (Hit-or-Miss + Reconstruction)
  - Template Matching
  - (Tuỳ chọn) Contour/Shape-based
## Hướng dẫn Cài đặt & Chạy thử nghiệm
### Yêu cầu hệ thống
.........
### Cách chạy dự án
......
