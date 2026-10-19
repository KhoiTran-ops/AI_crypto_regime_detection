# Spec: external-data
## Mục tiêu
Bổ sung biến còn thiếu trong blueprint; không tải OHLC/return BTC/ETH từ sàn khác để so với Binance.
FRED: VIX, Treasury 2Y/10Y, S&P500; Yahoo Finance: ICE DXY index; Coin Metrics: CapMrktCurUSD BTC/ETH; DefiLlama: tổng stablecoin USD và TVL.
CoinGecko dài hạn miễn phí bị giới hạn: dùng Coin Metrics với định nghĩa nguồn rõ ràng. Không gọi tổng vốn hóa BTC+ETH là tổng thị trường crypto.
## Hợp đồng dữ liệu
Long table: series, observation_date UTC, available_date UTC, value float64, unit, source.
Khóa series/observation_date duy nhất; xung đột trùng dừng pipeline. Không có vô cực, đơn vị chuẩn USD/index/percent.
FRED giá trị '.' giữ NA; không coi ngày nghỉ là lỗi. Biến close thị trường Mỹ chỉ dùng từ ngày làm việc Mỹ kế tiếp (giả định bảo thủ so với close cùng ngày); daily crypto aggregates dùng từ ngày UTC kế tiếp.
DXY chỉ nhận instrumentType INDEX và đúng symbol DX-Y.NYB; không thay bằng futures/broad USD mà giữ tên DXY.
Biến growth là log-change một ngày giữa hai quan sát liên tiếp; gap không nén. Nonpositive TVL không dùng log; ghi audit.
As-of join chỉ sau available_date, lưu tuổi quan sát; weekend chỉ giữ thông tin công bố gần nhất, không giả là return mới.
Archive hồi cứu không chứng nhận historical vintages. Đánh giá trễ thêm một tuần như timing sensitivity (một chu kỳ báo cáo tuần), không gọi là tối ưu.
## Chạy và nghiệm thu
python -m regime.cli external-data
Git giữ schema, source URLs và hash của snapshot chuẩn hóa; dữ liệu gốc/chuẩn hóa bên ngoài được tải cục bộ, không phân phối lại chuỗi SP500. Cài môi trường từ lock; không yêu cầu API key.
Mặc định dùng snapshot cục bộ nếu có; clone mới tự tải và kiểm tra hash chuẩn hóa đã công bố. Provider sửa lịch sử sẽ dừng thay vì âm thầm thay dữ liệu. Không ghi đè baseline.
