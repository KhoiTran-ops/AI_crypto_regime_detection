# Kế hoạch mở rộng theo blueprint
Ngày 19/10/2026; người dùng đã yêu cầu spec → plan → code ngay, không chờ duyệt giữa các phase.
1. Khóa invariants config/baseline và thẩm định khả năng nguồn.
2. Tải, normalize, audit, đóng gói ngoại sinh và provenance.
3. Tính covariates/liquidity/correlation theo lịch khả dụng.
4. Chạy TVTP macro/crypto và homogeneous matched-cohort.
5. Bổ sung HDBSCAN, BOCPD và MS-GARCH; validation trước test.
6. Xuất kết quả, danh mục và diagnostics riêng, không PDF.
7. Dashboard, README clone/replay, kiểm tra source hashes và test cốt lõi.
Rủi ro: API thay đổi/schema, publication vintages thiếu, composition stablecoin thay đổi, convergence và retrospective selection. Không giấu lỗi/thiếu dữ liệu; không thay dữ liệu thiếu bằng zero.


## Hợp nhất kết quả và dashboard · 19/10/2026
1. D01 — Gói kết quả theo nội dung, loại bản sao byte-identical; giữ nguyên mọi hash khoa học và 7 bảng tổng hợp toàn nghiên cứu. Kiểm tra round-trip từng artifact trước khi dọn file rời.
2. D02 — Menu trái 5 trang; thống nhất chọn tài sản/mô hình; biểu đồ điểm có mục tiêu rõ ràng, không trộn joint density khác biến hoặc gán mật độ cho clustering.
3. D03 — Data gom thị trường, phái sinh, vĩ mô, on-chain và phương pháp. UI không có nhật ký, nhãn bổ sung hay phần giới hạn.
4. D04 — Kiểm tra số liệu, toàn vẹn, tái lập clone, trình duyệt desktop/mobile; cập nhật README. Không fit lại, đổi mẫu, RMSE hoặc định nghĩa R.
