# ADR-002: Đổi khối lượng tuyệt đối thành khối lượng tương đối

Trạng thái: chấp nhận ngày 16/10/2026, trước khi mở holdout.

## Bằng chứng

Pilot dùng return, rv30 và ln(1+quote_volume). Trên train BTC 2017–2022, HMM hai trạng thái cho self-transition gần 1 và 0,9991. Một mẫu mô phỏng 1.902 ngày bắt đầu từ phân phối khởi tạo của mô hình chỉ thăm một trạng thái. Tăng số khởi tạo không tạo ra thông tin về trạng thái không được thăm.

Recovery pilot một khởi tạo thất bại cho K=2..5; đây cũng không đúng bộ ước lượng nhiều khởi tạo dùng trong nghiên cứu. Kết quả pilot và diagnostics không bị ghi đè, được giữ ở outputs/pilot_absolute_volume/ trên máy làm việc.

## Quyết định

Dùng volume_z: chuẩn hóa log quote volume hôm nay bằng trung bình và độ lệch chuẩn của 30 ngày trước. Biến này so hoạt động hôm nay với nền gần đây, giảm việc phân loại thời đại tăng trưởng quy mô sàn. Chạy lại toàn bộ lựa chọn HMM/GMM theo đặc tả mới trước test.

Recovery chính dùng 20 khởi tạo và chọn nghiệm hội tụ có likelihood train cao nhất. Monthly refit dùng 3 khởi tạo cố định để giới hạn chi phí, nên kiểm chứng 20-start không phải đảm bảo từng monthly refit đạt chất lượng tương đương. Learning curve báo riêng độ nhạy theo khởi tạo đơn.

## Hệ quả và giới hạn

Đây là sửa đặc tả dựa trên chẩn đoán train, không phải bằng chứng về hiệu quả giao dịch trên test. Volume_z vẫn có thể lệch phân phối và có outlier. Không so log density giữa đặc tả volume tuyệt đối và tương đối như cùng một target.
