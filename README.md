# Đề tài 1: Phát hiện dấu cộng/dấu X (Cross/X-marker) trên ảnh
## Ứng dụng Morphological Hit-or-Miss và Marked Features (Morphological Reconstruction)

## 👥 Thành viên
- Hồ Đức Nhân Thiện - 31231026999
- Trần Quốc Bảo - 31231022657
- Nguyễn Huỳnh Tấn Phát - 31231024397
- Nguyễn Ngọc Minh Đức - 31231020325

---

## 📖 Giới thiệu (Introduction)
Dự án này là Đồ án cuối kỳ môn Xử lý và Phân tích Hình ảnh. Mục tiêu của dự án là nghiên cứu, cài đặt và đánh giá hiệu quả của các phương pháp Hình thái học (Morphological) và Đặc trưng đánh dấu (Marked Features) để phát hiện và trích xuất các mẫu (pattern) cụ thể trong ảnh kỹ thuật số.

**Bài toán ứng dụng cụ thể:** Phát hiện dấu cộng/dấu X (cross/X-marker) xuất hiện trong ảnh (ví dụ: marker in trên giấy, ảnh chụp tài liệu, hoặc ảnh kỹ thuật).

---

## ✨ Tính năng chính (Features)
- Tiền xử lý ảnh (Khử nhiễu, binarize, cân bằng sáng).
- Phát hiện pattern sử dụng Morphological Hit-or-Miss Transform.
- Trích xuất cấu trúc phức tạp sử dụng Marked Features và Morphological Reconstruction.
- Giao diện Web tương tác trực quan cho phép người dùng upload ảnh và tinh chỉnh tham số theo thời gian thực.
- Bảng module so sánh hiệu năng (Accuracy, Processing Time) giữa các thuật toán.

---

## ✅ Checklist công việc theo thành viên
> Gợi ý: mỗi mục nên được tạo thành 1 GitHub Issue và làm qua Pull Request (PR).

### 1) Hồ Đức Nhân Thiện — Dataset & Annotation
- [ ] Lập quy chuẩn dữ liệu: thư mục, format ảnh, cách đặt tên.
- [ ] Thu thập/chụp ảnh (tối thiểu 200 ảnh, đa điều kiện sáng/góc/nền).
- [ ] Xây dựng tool gán nhãn (click để lưu tâm marker) hoặc hướng dẫn dùng tool ngoài.
- [ ] Gán nhãn ground truth (tọa độ tâm hoặc bbox) và kiểm tra chất lượng nhãn.
- [ ] Chia tập train/val/test + thống kê số lượng ảnh từng tập.
- [ ] Viết `docs/dataset.md`: mô tả cách thu thập & format nhãn.

### 2) Trần Quốc Bảo — Baseline & Template Matching
- [ ] Cài đặt baseline 1: Template Matching (multi-scale nếu có).
- [ ] Cài đặt baseline 2 (tuỳ chọn): contour/shape filtering.
- [ ] Chuẩn hoá output dự đoán: list điểm tâm + confidence (nếu có).
- [ ] Script chạy batch toàn bộ dataset cho baseline.
- [ ] Tổng hợp thời gian xử lý trung bình / ảnh.

### 3) Nguyễn Huỳnh Tấn Phát — Morphological Core (HMT + Reconstruction)
- [ ] Tiền xử lý (CLAHE/normalize, denoise, threshold Otsu/adaptive).
- [ ] Implement Morphological Hit-or-Miss cho dấu `+` và/hoặc `x`.
- [ ] Implement Marked Features + Morphological Reconstruction để refine vùng pattern.
- [ ] Post-filter (min area, min distance, loại nhiễu).
- [ ] Chuẩn hoá output dự đoán giống baseline.
- [ ] Viết mô tả thuật toán ngắn trong `docs/method.md`.

### 4) Nguyễn Minh Đức — Web App & Integration
- [ ] Thiết kế API backend (Flask/FastAPI): `/detect` nhận ảnh + params.
- [ ] Tạo UI web: upload ảnh, chọn thuật toán, chỉnh tham số.
- [ ] Hiển thị overlay kết quả (marker/bbox) + bảng thông số (count, time).
- [ ] Tích hợp chạy 3 mode: morphological / template / contour.
- [ ] Viết `docs/how-to-run.md` + script chạy local.

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

## 🧪 Đánh giá & So sánh (gợi ý)
- Metrics: Precision, Recall, F1-score (theo khoảng cách tâm), Processing Time.
- So sánh tối thiểu:
  - Morphology (Hit-or-Miss + Reconstruction)
  - Template Matching
  - (Tuỳ chọn) Contour/Shape-based
