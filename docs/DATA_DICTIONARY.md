# Dữ liệu và từ điển biến

Snapshot nghiên cứu: 30/09/2026 UTC. Chỉ Binance, nguồn công khai không cần API key.

| Tập dữ liệu | Định nghĩa | Lịch sử dùng | Xử lý |
|---|---|---|---|
| Spot BTCUSDT, ETHUSDT | Kline ngày, giá theo USDT | Raw 17/08/2017–30/09/2026 | Archive tháng, SHA-256 từ CHECKSUM |
| Funding USD-M perpetual | Tổng tỷ lệ settlement thực có trong ngày UTC | BTC từ 10/09/2019; ETH từ 27/11/2019 | Giữ sự kiện gốc, cộng theo ngày; không áp đặt 3 settlement/ngày |
| OI USD-M perpetual | sum_open_interest, sum_open_interest_value | Mẫu chung từ 01/12/2021 | Snapshot hợp lệ cuối ngày, cách nửa đêm không quá 15 phút |

Archive 1 phút ngày đầu của cả hai cặp bắt đầu 04:00 UTC, chỉ có 1.200 phút. Bằng chứng và checksum: outputs/data_audit/start_evidence.json. Vì vậy ngày đầy đủ đầu tiên dùng để tạo biến là 18/08/2017. Ngày 17/08 vẫn giữ nguyên trong raw.

| Biến | Công thức | Đơn vị / thông tin khả dụng |
|---|---|---|
| open, high, low, close | OHLC spot ngày UTC | USDT / BTC hoặc ETH |
| base_volume | Lượng tài sản giao dịch spot | BTC / ETH |
| quote_volume | Quote asset volume | USDT |
| return | ln(close_t / close_t-1) | Log return ngày, không nhân 100 |
| rv7, rv30, rv60 | sd(return, ddof=1, cửa sổ 7/30/60) × sqrt(365) | Volatility năm hóa, quan sát đến cuối t |
| log_volume | ln(1 + quote_volume) | Biến trung gian, không còn là feature chính |
| volume_z | (log_volume_t − mean(log_volume_t-30:t-1)) / sd(t-30:t-1) | Khối lượng tương đối so với 30 ngày trước |
| illiquidity | abs(log return) / quote_volume | Proxy theo log return, không phải Amihud nguyên bản |
| range | ln(high / low) | Proxy biên độ trong ngày |
| funding | Tổng fundingRate của các sự kiện ngày UTC | Tỷ lệ, ví dụ 0,0003 = 0,03% |
| funding_count | Số settlement quan sát được | Số sự kiện; ngày đầu có thể không đủ |
| oi | OI snapshot hợp lệ cuối ngày | Đơn vị tài sản hợp đồng theo trường nguồn |
| oi_quote | OI quy đổi tại snapshot | USDT |
| oi_change | ln(oi_t / oi_t-1) | Thiếu ở cả ngày lỗi và ngày liền sau |
| funding_lag2, volume_lag2 | Biến trễ hai ngày | Dùng trong transition regression |

## Quy tắc chất lượng

- Reindex lịch UTC đầy đủ; không forward-fill giá hoặc OI, không thay thiếu bằng 0.
- Không tính return bắc qua ngày thiếu. Rolling window phải đủ số quan sát.
- Kiểm tra OHLC dương và high/low bao phủ open/close; volume không âm; duplicate xung đột dừng pipeline.
- Spot từ 01/01/2025 sử dụng microseconds; dữ liệu trước đó milliseconds. Chuẩn hóa trước khi ghép.
- ZIP chỉ đọc CSV, không giải nén đường dẫn tùy ý; giới hạn kích thước tải/giải nén.
- Funding: duplicate timestamp phải đồng nhất; gap trên 12 giờ khiến các ngày liên quan thiếu.
- OI: loại snapshot không dương; nếu snapshot cuối hợp lệ quá cũ thì để thiếu cả ngày. Lưu số snapshot bị loại.
- HMM dùng các đoạn ngày liên tục như các chuỗi riêng; không nén khoảng trống thành một bước ngày.
- Quy trình econometric yêu cầu chuỗi liên tục; dừng nếu bị thiếu ngày.
- Tất cả thời gian là UTC. Spot và perpetual là hai thị trường khác nhau.

## Nguồn chính thức

[Binance public data và cấu trúc archive](https://github.com/binance/binance-public-data) mô tả định dạng, CHECKSUM và thay đổi timestamp.
[Binance USD-M market data](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data) mô tả API funding.
Dấu vết tải: data/manifest_*.json và data/manifest_archives.csv và outputs/data_audit/. Hash snapshot xử lý: results/protocol_lock.json.

Archive tải hồi cứu có thể khác phiên bản từng được công bố trước đây. Pipeline xử lý nhân quả và độ trễ giao dịch không khôi phục được các phiên bản lịch sử chưa lưu.
