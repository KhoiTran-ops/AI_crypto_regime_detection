# Tasks mở rộng
- [x] E01 Đối chiếu blueprint, spec, plan và nguồn truy cập thực.
- [x] E02 Dữ liệu chuẩn hóa, source log, snapshot/hash và replay.
- [x] E03 Feature nhân quả, liquidity/correlation, availability audit.
- [x] E04 TVTP macro/crypto cùng đối chứng khớp mẫu.
- [x] E05 HDBSCAN và BOCPD causal ngoài mẫu.
- [x] E06 MS-GARCH independent variance và sanity.
- [x] E07 Kết quả/danh mục/độ nhạy và dashboard.
- [x] E08 Clone sạch, test gọn, kiểm chứng invariants, bàn giao.

Kết quả: 30 cấu hình bổ sung cho hai tài sản, 638 ngày/cấu hình; tất cả refit cuối hội tụ. HDBSCAN_DENSITY chọn min_cluster_size BTC17/ETH23 trên train; các cấu hình thô thất bại vẫn giữ.

E08 đạt: clone sạch cài 63 pin, 14 tests, tải lại nguồn/khớp hash; verify 141 artifacts gốc +189 artifacts mở rộng/14 inputs; UI BTC/ETH và Edge desktop/mobile đã qua. Không sinh PDF; chưa push.


## Dashboard và kết quả thống nhất
- [x] D01 — Gói kết quả và bảng chung, kiểm tra nguyên vẹn.
- [x] D02 — Dashboard thống nhất và menu trái.
- [x] D03 — Data và phương pháp chung, bố cục theo reference.
- [x] D04 — QA, clone, tài liệu và bàn giao.

D04 đạt: clone sạch không cache/raw, 15 tests; verify/extend không làm thay đổi working tree; UI BTC/ETH và download control khi chưa có ngoại sinh hoạt động. Edge 320–1440 px; return chart 19, joint-core 2. Không fit lại hoặc đổi mẫu/RMSE/R.


## Nhãn trạng thái và bản tổng hợp nghiên cứu · 19.10.2026
- [x] S01 — Kiểm tra cơ sở chia mẫu bằng tài liệu gốc và nested origins đã có.
- [x] S02 — Nhãn thị trường cho HMM chính BTC/ETH, giữ nguyên mã và số liệu.
- [x] S03 — Một HTML độc lập: bảng/hình, dữ liệu/biến/tham số, phương pháp, kết luận.
- [x] S04 — Xác minh kết quả nguyên vẹn, kiểm thử gọn và kiểm tra trình duyệt.
Lịch tiếp tục một lần: 18:00 Asia/Saigon, 19.10.2026; bỏ qua nếu hoàn tất.
Phạm vi: không đổi mẫu, K, ngưỡng RMSE, kết quả mô hình; không PDF, không push.

S04 đạt: HTML độc lập 19 bảng/5 hình SVG; Edge desktop/mobile và tải đúng byte; bản sao không raw/cache tạo lại byte-identical; 16 tests, ruff và verify 141+189 artifacts/14 inputs đạt. Chỉ thêm 1 test ngắn chống gán nhầm nhãn. Không fit lại/đổi mẫu/ngưỡng/PDF/push.
Lịch 18:00 được kích hoạt nhưng thất bại (exit 1) do script UTF-8 không BOM với Windows PowerShell. Đã sửa mã hóa và kiểm tra parser; không khẳng định đã kiểm chứng chạy continuation thành công. Công việc tiếp tục theo yêu cầu trực tiếp của người dùng và đã hoàn tất.
