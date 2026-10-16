# ADR-001: Snapshot dữ liệu và thiết kế đánh giá

Trạng thái: chấp nhận ngày 16/10/2026, trước đánh giá holdout.

## Bối cảnh

Cần kết quả nghiên cứu có thể tái lập và dashboard tiếng Việt. Người dùng chọn Binance, nguồn miễn phí và mốc 2017, yêu cầu cơ sở chọn mẫu. Không dùng tỷ lệ train/test tùy ý để khẳng định độ đủ mẫu.

## Quyết định

Dùng toàn lịch sử BTCUSDT và ETHUSDT spot, raw từ 17/08/2017, loại ngày đầu không đủ khỏi tính feature. Analysis bắt đầu 17/10/2017 sau warm-up 60 return, dù đặc tả chính dùng rv30. Cutoff 30/09/2026 là tháng hoàn chỉnh trước khi thiết kế, không phải điểm gãy kinh tế được ước lượng.

Mẫu A: train 2017–2022 (1.902 ngày dùng được), validation 2023–2024, holdout 2025–30/09/2026. Mốc lịch là lựa chọn thiết kế, được đối chiếu bằng learning curve và outer origins 2020–2024 có lựa chọn K chỉ từ quá khứ.

HMM Gaussian covariance diagonal có số tham số (K−1)+K(K−1)+2Kd. Lưới K=2..5. Chọn điểm validation cao nhất trong các ứng viên đạt recovery simulation: trung bình RMSE toàn ma trận transition ≤0,05; RMSE tương đối sigma return ≤20%; hội tụ ≥90%. Các mức này là tiêu chuẩn do dự án khai báo, không phải định lý về cỡ mẫu.

Recovery dùng 100 mẫu từ Gaussian DGP ước lượng trên train, mỗi mẫu fit nhiều khởi tạo như bộ ước lượng thật. BTC chọn K=4, ETH K=5; BTC K=5 bị loại bởi recovery. Heavy-tail t5 được báo riêng, không dùng để lén đổi ngưỡng. Các mức sai số có tính điều kiện theo DGP; không chứng minh thị trường có đúng K trạng thái.

Mô hình đối chứng: GMM, K-means, HMM return-only, Markov Switching AR(1). Mở rộng: HMM thêm funding/OI và Markov Switching với transition phụ thuộc funding/volume trễ hai ngày. Không nhận các mô hình gợi ý khác trong blueprint như những hạng mục đã triển khai.

## Các lựa chọn khác

Chia ngẫu nhiên bị loại vì làm lẫn quá khứ/tương lai. Áp nhãn bull/bear/stress trước ước lượng bị loại vì không có nhãn thật. Dùng toàn mẫu để vừa chọn vừa chấm bị loại vì thiên lệch đánh giá.

Các biến VIX/DXY và nguồn khác bị loại theo quyết định chỉ dùng Binance. Funding/OI không được kéo lùi về 2017 khi chưa tồn tại. Mẫu C ngắn hơn được coi là khám phá, không tuyên bố đã đạt recovery như A.

## Hệ quả

Không khẳng định mốc cuối 2022 là tối ưu hoặc mọi trạng thái đều có độ chính xác như nhau. Báo đầy đủ kết quả nhiều origin, seed sensitivity, độ phủ và giới hạn. Mọi lựa chọn được ghi hash trước test. Tái ước lượng hàng tháng chỉ dùng ngày trước origin; test trước đó được bổ sung vào lịch sử theo quy tắc cố định.
