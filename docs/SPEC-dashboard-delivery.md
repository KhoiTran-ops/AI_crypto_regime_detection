# Spec: dashboard-delivery
Thêm trang mở rộng: coverage, series, models/coefficients, liquidity/correlation, portfolio và source limitations. Hiển thị baseline và extension riêng.
Clone repo + Python3.12 + requirements.lock mở được kết quả không tải mạng/fit. Lệnh external-data/extend chạy lại bằng frozen snapshots.
Kiểm chứng clone sạch không raw/.venv; hash nguồn vào/ra, replay dữ liệu, một số test nhỏ chống leakage/gap và sanity probabilistic models.
Giữ test ít và có giá trị: một tệp extension tests gọn, không snapshot-test UI lớn. Không sửa/xóa core tests để làm xanh.
