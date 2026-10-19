# Kết quả nghiên cứu

- `tables/`: 7 bảng CSV tổng hợp cả BTC/ETH và toàn bộ 48 cấu hình mô hình, gồm điểm mô hình, chênh lệch điểm, danh mục, bất định, nguồn dữ liệu, cỡ mẫu và đặc trưng trạng thái. Ô trống là chỉ tiêu không áp dụng hoặc không được ước lượng, không phải 0.
- `research.zip`: dự báo validation/test, đường danh mục, hệ số, chẩn đoán, lựa chọn tham số và protocol; chỉ lưu một bản cho mỗi nội dung có cùng SHA-256. `index.json` ánh xạ tên logic sang `objects/<sha256>`. Giữ nguyên từng byte của kết quả đã công bố và các manifest cũ.

Dashboard đọc trực tiếp gói này. `python -m regime.cli verify-extensions` kiểm tra cả hai bộ kết quả và code/inputs đã khóa. Các lệnh tính nghiên cứu tự giải gói trước khi chạy, sau thành công tự đóng gói lại; không đổi công thức hay tham số. Cache dữ liệu ngoại sinh ở `extensions/*_features.parquet` và `external_panel.parquet` không vào Git.

Ví dụ đọc dự báo: `from regime.results import Results; Results().frame("BTCUSDT_HMM.parquet")`. Với một mô hình bất kỳ dùng `store.frame(store.model_path("BTCUSDT", "TVTP_MACRO"))`.


`tom_tat_ket_qua.html` là bản tổng hợp một file mở offline (bảng và 5 hình SVG nhúng). Tạo lại bằng `python -m regime.cli report`, không cần raw/cache hoặc fit lại. Bản này đọc kho đã chốt và không thay số liệu/cấu hình khoa học.
