# Spec: extended-models
## Mục tiêu
Giữ các mô hình gốc. Bổ sung TVTP macro và crypto cùng đối chứng homogeneous trên đúng mẫu; HDBSCAN, Bayesian online change-point và independent-switching GARCH(1,1).
Tái sử dụng statsmodels/NumPy/SciPy; chỉ thêm hdbscan cho approximate_predict chính thức, numba nếu cần tăng tốc likelihood có kiểm tra tương đương.
## Thiết kế
Giữ nguyên train/validation/test config.json và clock giao dịch. Mọi fit/StandardScaler chỉ dùng lịch sử trước origin. Mẫu mở rộng trim theo ngày dữ liệu thực đủ, công bố số ngày. Không kéo phái sinh/TVL về 2017.
TVTP hai trạng thái để khớp MS0 hiện có; covariates trễ 2 ngày theo clock hiện tại. Funding/volume là đối chứng, macro gồm VIX, DXY, Treasury và SP500; crypto gồm cap/stablecoin/TVL/correlation, dùng nhóm nhỏ có ý nghĩa.
Amihud proxy dùng |simple return| / quote-volume USDT; không nhận là bid-ask spread. Cửa sổ 30 ngày kế thừa rv30 của nghiên cứu, không tối ưu theo test.
HDBSCAN: OOS approximate_predict, giữ noise=-1 và membership strength; không gọi strength là xác suất hiệu chỉnh hay noise là stress. Tham số cấu trúc lấy từ kích thước trạng thái train/k lưới đã có; selection EOM mặc định tài liệu. Báo tất cả ứng viên và chọn bằng validation, không threshold tùy tiện.
BOCPD Normal-Inverse-Gamma conjugate Student-t predictive theo Adams & MacKay (2007); hazard từ thời gian duy trì train HMM, báo nhạy theo các trạng thái. Không dùng nhãn stress giả.
MS-GARCH: Haas/Mittnik/Paolella (2004), parallel regime-specific GARCH(1,1), common mean; alpha,beta>=0, alpha+beta<1 bằng softmax; transition row stochastic. Hai trạng thái khớp Markov comparator; fit deterministic nhiều khởi tạo và công khai hội tụ.
Giới hạn tối ưu là numerical settings, không phải economic thresholds; ghi sensitivity khi hội tụ chưa chắc chắn.
