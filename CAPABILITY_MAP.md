# Mở rộng dữ liệu và đối chứng
Phạm vi được người dùng yêu cầu triển khai ngay ngày 19/10/2026.

| Module | Trách nhiệm | Phụ thuộc |
|---|---|---|
| external-data | Nguồn bổ sung, chuẩn hóa, lịch khả dụng, snapshot/hash | Binance snapshot hiện có |
| extended-models | TVTP macro/crypto, liquidity, correlation, đối chứng còn thiếu | external-data |
| research-results | Lựa chọn validation, đánh giá test, danh mục, diagnostics | extended-models |
| dashboard-delivery | Dashboard đọc snapshot, clone và chạy, kiểm chứng | research-results |

Thứ tự: external-data → extended-models → research-results → dashboard-delivery.
Không thay config.json, mốc chia mẫu hoặc ngưỡng recovery 0,05. Không sinh PDF.
Bayesian HMM, RL và mở rộng SOL/BNB nằm trong mục tùy chọn 4.3/3.1 của blueprint, không phải điều kiện nghiệm thu lượt này.
