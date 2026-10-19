# Phương pháp và cách đọc kết quả

## Câu hỏi nghiên cứu đã thu hẹp

1. Với đặc tả cố định, K nào trong 2–5 đạt cả điểm validation và điều kiện recovery?
2. Các trạng thái khác nhau về return, volatility và độ bền như thế nào?
3. Funding/volume trễ có liên quan đến transition và cải thiện mật độ return ngoài mẫu không?
4. HMM so với GMM/K-means và Markov Switching như thế nào trên cùng lịch?
5. Giảm tỷ trọng theo xác suất trạng thái sigma cao nhất có cải thiện volatility targeting sau chi phí không?

Không nghiên cứu tác động macro vì phạm vi Binance-only. Không coi mô hình nhận diện được nhãn bull/bear thật.

## Mô hình và clock

Feature chính: log return, rv30, volume_z. Chuẩn hóa StandardScaler chỉ fit train trước origin. Covariance diagonal; EM tối đa 300 vòng, tol 0,01, kiểm tra mức tăng likelihood cuối thay vì chỉ tin cờ đã chạm số vòng tối đa. Khởi tạo đầu kỳ 20 seeds; refit hàng tháng 3 seeds.

Lọc forward cho P(S_t | Y_1:t), không dùng smoothed probabilities cho giao dịch. Nhãn đầu tiên xếp theo sigma return; các refit tiếp theo ghép Hungarian theo mean/SD raw để giảm đảo nhãn. Trạng thái sigma cao nhất được xác định lại từ mô hình fit quá khứ, không giả định luôn là cùng mã R.

Mật độ dự báo return dùng prior trước quan sát Y_t. Mật độ joint gồm toàn bộ biến ngày t và Jacobian chuẩn hóa. Chỉ so trực tiếp joint score khi biến và đơn vị giống nhau. K-means không có log density.

## Mẫu mở rộng

B: funding chung từ 28/11/2019; validation 2023–2024. C: log-change OI chung từ 02/12/2021; train đến 2023, validation 2024. K=2 hoặc 3, chọn bằng validation trong từng cohort. Giữ cùng holdout 2025–30/09/2026, ghi rõ số ngày hợp lệ.

NHHMM ở đây là Gaussian Markov Switching hai trạng thái có transition logit theo funding và volume_z trễ 2 ngày. Đối chứng MS0 dùng đúng các ngày và train history của cùng cohort B. Hệ số theo raw state; không đọc dấu hệ số nếu chưa đối chiếu trạng thái đích và mã sigma cao nhất.

## Kinh tế và bất định

Tỷ trọng volatility targeting: min(1, 0,4/rv30). Tỷ trọng regime-aware: tỷ trọng volatility targeting nhân (1−P_high_vol). Không tối ưu ngưỡng theo test. Tín hiệu cuối ngày t khớp tại open t+2; giữ tiền mặt hai ngày đầu, thanh lý tại open ngày cuối. Các phương án dùng cùng clock. Giá cuối để tính P&L là open 30/09/2026, còn trạng thái cuối dùng thông tin close 30/09/2026.

Tài sản/cash được cập nhật theo biến động giá trước khi tái cân bằng, chi phí tính trên giá trị giao dịch thật và giải điều chỉnh tỷ trọng sau phí. Kịch bản 5/10/25 bps mỗi chiều, 10 bps chính; không giả định đây là biểu phí thực của Binance.

Return dùng trong đặc trưng là log return; P&L dùng return tài khoản thực. Năm hóa 365; cash return 0. CAGR dựa trên số khoảng ngày thực; Sharpe dùng trung bình ngày/SD ngày × sqrt(365).

Bootstrap paired circular blocks giữ cùng ngày giữa mô hình; 1.000 lần, block 30 ngày chính, 7/60 cho độ nhạy Sharpe. Không coi 638 ngày là 638 thử nghiệm độc lập. Khoảng tin cậy là xấp xỉ, không hiệu chỉnh nhiều so sánh; chỉ số Sharpe mô phỏng không phải cam kết tương lai.

## Căn cứ phương pháp

- [Hamilton (1989), Econometrica](https://doi.org/10.2307/1912559): cơ sở Markov Switching.
- [Pohle et al. (2017), bản tác giả](https://arxiv.org/abs/1701.08673): chọn số trạng thái cần xem diễn giải và sai đặc tả, không chỉ tiêu chí thông tin.
- [Cerqueira, Torgo & Mozetič (2020), bản tác giả](https://arxiv.org/abs/1905.11744): đánh giá dự báo chuỗi thời gian và ảnh hưởng của phương pháp ước lượng hiệu năng.
- [Morris, White & Crowther (2019)](https://doi.org/10.1002/sim.8086): thiết kế và báo cáo mô phỏng theo mục tiêu, DGP, estimand, method, performance.
- [hmmlearn tutorial](https://hmmlearn.readthedocs.io/en/stable/tutorial.html): nhiều khởi tạo và vấn đề local optima.
- [statsmodels MarkovAutoregression](https://www.statsmodels.org/stable/generated/statsmodels.tsa.regime_switching.markov_autoregression.MarkovAutoregression.html): AR switching và TVTP.

Các tài liệu này hỗ trợ phương pháp, không cung cấp căn cứ máy móc cho đúng ngày cuối train/validation của dự án. Tài liệu phần mềm và tham số được đối chiếu ngày 16/10/2026.

## Diễn giải chính xác phạm vi kiểm chứng

Ngưỡng 0,05 của recovery là RMSE toàn ma trận transition, không phải giới hạn sai số của từng p_ii. Không suy ngược từ việc qua ngưỡng trung bình sang việc từng persistence/duration đã đạt độ chính xác đó. Các khai báo trước là hồ sơ nội bộ có hash, không phải public preregistration.

Sharpe/SD dùng toàn chuỗi net_return đã lưu, bao gồm giá trị ban đầu bằng 0 trên calendar chung; CAGR dùng số khoảng ngày len−1. Tái lập đã kiểm chứng trong môi trường Windows được khóa; không nhận là lần cài lạnh thứ hai hoặc bảo đảm kết quả byte-identical trên hệ khác.


## Cơ sở chia mẫu · rà soát 19.10.2026

**Có thể giải thích thiết kế hiện tại, chưa thể khẳng định tối ưu ngày cắt.**
- Raw Binance bắt đầu 17.08.2017; ngày đầu không trọn vẹn bị loại. Mốc 17.10.2017 đồng bộ panel có rolling 60 ngày; riêng ba biến HMM không bắt buộc toàn bộ 60 ngày.
- Train 17.10.2017–31.12.2022 có 1.902 ngày; validation 2023–2024 có 731; test 01.01.2025–30.09.2026 có 638. Đây là phân chia theo thời gian, không phải quy tắc 60/20/20. Mỗi origin chỉ dùng dữ liệu đã xảy ra, theo [Hyndman & Athanasopoulos, §5.10](https://otexts.com/fpp3/toolbox.html#time-series-cross-validation). [Cawley & Talbot (2010)](https://jmlr.org/papers/v11/cawley10a.html) giải thích sai lệch khi chọn và đánh giá mô hình trên cùng dữ liệu.
- Về kinh tế, train bao trùm crypto winter khởi đầu tháng 11/2021 và suy giảm năm 2022 theo [ECB, 23.06.2023](https://www.ecb.europa.eu/press/key/date/2023/html/ecb.sp230623_1~80751450e6.en.html). Validation chứa thay đổi thể chế quan trọng: [SEC phê duyệt spot Bitcoin ETP ngày 10.01.2024](https://www.sec.gov/newsroom/speeches-statements/gensler-statement-spot-bitcoin-011023). Các sự kiện hỗ trợ tính đa dạng bối cảnh, **không suy ra** 31.12.2022 hoặc 31.12.2024 là điểm gãy hay mốc tối ưu. Chưa có chứng cứ các mốc được đăng ký công khai trước khi dữ liệu phát sinh.
- Kiểm tra nested origins đã lưu: BTC chọn K=4,3,3,5,5 và ETH K=5,3,5,5,4 cho các năm outer 2020–2024. K nhạy theo thời kỳ; kiểm tra này không áp recovery gate giống hệt lựa chọn cuối nên không đọc như xác nhận BTC K=4/ETH K=5 ở mọi origin.
- Sau chọn K, fit đầu test dùng train+validation (2.633 ngày đến 31.12.2024), rồi mở rộng mỗi tháng. Funding/OI và biến ngoại sinh có cohort riêng theo ngày khả dụng, không lấp quá khứ trước khi nguồn tồn tại. Các mở rộng đã xem test là phân tích hồi cứu.
- Chọn BTC/ETH là phạm vi hai tài sản có lịch sử dài/liên tục ở Binance, không phải mẫu ngẫu nhiên đại diện toàn thị trường crypto. Ngày kết thúc là snapshot hoàn chỉnh 30.09.2026. Cỡ mẫu lịch không đồng nghĩa số quan sát độc lập; learning curve và recovery chưa chứng minh một cỡ mẫu tối thiểu phổ quát.

## Nhãn thị trường HMM chính
Nhãn hiển thị lấy từ hồ sơ đầu test đến 31.12.2024, dùng lợi suất trung bình và thứ bậc biến động tương đối theo từng tài sản. Không áp nhãn HMM cho mã R của mô hình khác. Nhãn BTC/ETH cùng R1/R2/R3 lần lượt: thiên tăng ít biến động, trung tính, thiên tăng biến động cao; BTC R4 và ETH R5 là biến động rất cao; ETH R4 thiên giảm biến động cao. Hồ sơ trọng số trong bảng khác với mean/sigma emission dùng cho bootstrap tham số. CI mean emission ETH R3/R4 chứa 0 nên nhãn hướng là mô tả, không phải kết luận dấu lợi suất có ý nghĩa thống kê. Hungarian giảm đảo mã, không bảo đảm ý nghĩa kinh tế bất biến sau mọi refit.
Thuật ngữ tham khảo [Koki et al. (2022)](https://arxiv.org/html/2011.03741) và [Cortese et al. (2023)](https://link.springer.com/article/10.1007/s42521-023-00085-x); các tài liệu này không xác nhận trực tiếp nhãn của mô hình dự án.
