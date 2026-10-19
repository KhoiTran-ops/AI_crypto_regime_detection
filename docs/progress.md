## Hợp nhất kết quả và dashboard · 19/10/2026

316 artifact được bảo toàn từng byte trong gói content-addressed, 224 nội dung duy nhất (loại 92 bản sao). Kết quả công bố gồm 7 bảng CSV + research.zip: 10,82 → 5,49 MB. Dashboard 5 trang thống nhất; 24 cấu hình/tài sản, biểu đồ return mặc định 19 mô hình, joint-core 2; Data gom mọi nhóm nguồn và phương pháp. Không đổi mẫu/RMSE/R hay fit lại. Kiểm tra: 15 tests, 62 tổ hợp AppTest, Edge desktop/mobile, hash khoa học và summary khớp. Clone sạch không raw/cache chạy verify/extend/15 tests và mọi trang BTC/ETH; working tree giữ sạch. Chi tiết QA: docs/dashboard_qa.json, docs/results_packaging_qa.json.

> Cập nhật 19/10/2026: đã mở rộng ngoài Binance-only với 30 cấu hình bổ sung. Xem [task mới](../tasks/todo.md), [phương pháp mở rộng](EXTERNAL_METHODS.md) và [QA](extension_qa.json). Nội dung 16/10 bên dưới được giữ như lịch sử bản gốc; bài/PDF cũ chưa cập nhật.

# Tiến trình triển khai và bàn giao

Cập nhật 16/10/2026. Code nằm tại F:\AI_crypto\AI_crypto_regime_detection; remote GitHub đã cấu hình. Chưa push.

| Hạng mục kế hoạch | Trạng thái và bằng chứng |
|---|---|
| T01 Workspace | Code đã ghi đúng folder được chỉ định. Chưa xác minh thao tác gắn folder vào UI project; phiên làm việc vẫn có cwd mirror. |
| T02 Nền tảng | 9 tham khảo có metadata/DOI xác minh, gồm 6 mục blueprint; literature_matrix.csv, BibTeX. Phần lớn nghiên cứu crypto chỉ đọc abstract/preview, không nhận là systematic review toàn văn. |
| T03 Môi trường | Python 3.12 riêng, requirements.lock, config, lệnh CLI; không cần API key. |
| T04–T07 Spot và mốc mẫu | 220 archive tháng, checksum; mỗi tài sản 3.332/3.332 ngày. Archive phút đầu xác minh 04:00 UTC ngày 17/08/2017; bỏ ngày không đủ khỏi feature. |
| T08 Funding | BTC 7.733, ETH 7.499 settlement; giữ lịch thật, tổng theo UTC, không lấp 0 giả. |
| T09 OI | 1.765 ngày/tài sản; 1.763 EOD hợp lệ. Loại 475 snapshot BTC và 218 ETH không dương; giữ ngày EOD thiếu. |
| T10 Macro | Ngoài phạm vi Binance-only theo quyết định người dùng. |
| T11 Ghép mẫu | A/B/C có mốc riêng, không kéo phái sinh về 2017. Xử lý nhân quả trên snapshot; historical vintages/publication clock không thể xác minh đầy đủ bằng archive hiện có. |
| T12–T13 Feature/HMM | Prefix-invariant feature, phân đoạn ngày thiếu, scaler train-only, forward filtering; không dùng full-sample smoothing cho giao dịch. |
| T13a Cỡ mẫu | Recovery 100 lần/ứng viên, 20 khởi tạo; heavy-tail 50; learning curve; nested origins 2020–2024; bootstrap tham số 100 lần/tài sản. BTC4/ETH5 qua ngưỡng điểm của dự án; BTC5 bị loại. MCSE và giới hạn công khai. |
| T14 Đối chứng | GMM, K-means, HMM return-only, MSAR; không so joint density khác biến. |
| T15 Mở rộng | HMM funding/OI và hai trạng thái TVTP theo funding/volume trễ hai ngày; hệ số/CI, đối chứng cùng cohort và sanity TVTP trên chuỗi có transition biết trước đã đạt. B/C vẫn là khám phá, chưa recovery toàn diện. |
| T16 Khóa | v1 protocol khóa trước test; v2 sửa RNG sau audit, giữ nguyên lựa chọn validation. Không quảng bá v2 như holdout mới untouched. |
| T17 Danh mục | Long/cash, volatility target 40%, signal close t/trade open t+2, cost 5/10/25 bps, hạch toán drift/entry/exit. |
| T18 Test | 638 ngày/tài sản, 9 mô hình/tài sản; có bootstrap paired blocks 30 ngày và Sharpe sensitivity 7/60. v2 đã hoàn tất; mọi refit cuối đều hội tụ và kết quả đã đối chiếu độc lập. |
| T19–T20 Bài/bảng/hình | Bản thảo đầy đủ tiếng Việt, PDF, BibTeX, hình PNG/PDF và bảng. v2 đã đồng bộ; PDF 13 trang được render và kiểm tra trực quan, không có chữ vượt biên. |
| T21–T22 Dashboard | 7 tab, BTC/ETH, dữ liệu/model/transition/derivatives/portfolio/methods. v2 đã qua Edge desktop/mobile, mọi tab BTC/ETH, tải PDF và đổi mức chi phí; không có lỗi console/mạng hoặc tràn ngang. |
| T23 Tái lập | v1 audit phát hiện RNG không được kiểm soát và dừng đúng; đã sửa, có regression test. v2 đạt: 3.752 archive kiểm tra checksum, sáu dataset tái dựng khớp, refit sạch 18 mô hình; sai khác dự báo lớn nhất 0,0. Bản Git archive sạch đã xác minh đủ 141 hash artifact và code khoa học, không cần raw để mở dashboard. |

## Kết quả và quyết định đã có

- Mẫu A: train 1.902, validation 731, test 638 ngày. C có OI: train 758/760 ngày, validation 364/366; test đủ 638.
- Feature chính: return, rv30, volume_z. Pilot volume tuyệt đối bị loại do trạng thái gần hấp thụ theo tăng trưởng sàn, có lưu bằng chứng.
- BTC K=4: transition RMSE 0,0415, MCSE 0,0102; ETH K=5: 0,0362, MCSE 0,0085. Chỉ là recovery có điều kiện theo DGP Gaussian, không chứng minh cỡ mẫu tối thiểu.
- HMM đổi trạng thái ít hơn clustering và có joint score cao hơn GMM trong test. Không suy từ đó ra ưu thế dự báo return hay lợi nhuận.
- Regime-aware HMM tại 10 bps: BTC CAGR -10,25%, volatility targeting -9,27%; ETH -12,31% so với -7,18%. Báo cả kết quả bất lợi.
- Chín kiểm thử lõi trong một tệp; kiểm tra mã đạt. Script thử trình duyệt chạy một lần không được giữ trong codebase.
- Full raw, runtime và cache không đưa vào Git. Chỉ code, dữ liệu kết quả gọn, manifest và hồ sơ tái lập.

## Vướng mắc đã xử lý

1. Máy ban đầu không có Python khả dụng: tạo môi trường riêng trong repository.
2. Timestamp pandas 3 có thể là microseconds: sửa so sánh khoảng ngày, thêm kiểm thử hồi quy.
3. Funding calendar UTC: sửa giới hạn cuối ngày, kiểm thử mốc UTC.
4. OI có snapshot lỗi trong ngày: áp quy tắc EOD hợp lệ đã đặt; không vứt cả ngày khi vẫn còn snapshot hợp lệ và đủ mới.
5. Một khởi tạo không đại diện bộ ước lượng 20-start: sửa thiết kế recovery, giữ pilot thất bại.
6. statsmodels 0.15 dùng Generator riêng: truyền rng=11; lưu v1 và công khai việc sửa sau lượt test đầu.
7. Hash qua Git có thể sai vì newline: nguồn/config chuẩn LF, snapshot artifact giữ byte bằng .gitattributes.

## Giới hạn không được che giấu

Archive hồi cứu; không có ground-truth regime; Gaussian diagonal và rolling feature là xấp xỉ; MCSE còn đáng kể; ETH chọn K chạm biên lưới; mẫu phái sinh ngắn hơn; chi phí không mô phỏng order book. Chưa có prospective trading hoặc triển khai hosting. Chưa có mẫu tạp chí để nghiệm thu submission-specific formatting.

## Sản phẩm đã có

- paper/bao_cao_nghien_cuu.pdf: 13 trang; nguồn Markdown và references.bib cùng folder.
- results/: 18 mô hình, bảng thống kê, ước lượng, bootstrap và danh mục; dùng chung với dashboard.
- docs/reproduction.json, browser_qa.json, pdf_qa.json, tvtp_sanity.json: bằng chứng kiểm chứng.
- docs/history/: protocol và bảng v1; đã khôi phục đúng byte và xác minh 9/9 tệp code khoa học v1 theo hash.

## Nghiệm thu cuối

Đã hoàn tất sản phẩm nghiên cứu và dashboard cục bộ; không còn lỗi kỹ thuật đang chặn. Working tree sẽ chốt sạch trong commit bàn giao. Chưa push/hosting. T01 về gắn folder trong UI và giới hạn historical vintages vẫn được ghi đúng phạm vi, không trình bày là đã xác minh.


## Bản tổng hợp và nhãn · 19.10.2026
Hoàn tất một HTML độc lập từ kho kết quả đã chốt: 19 bảng, 5 hình SVG nhúng; ba phần dữ liệu/biến/tham số, phương pháp, kết quả/kết luận. Cơ sở chia mẫu bổ sung tài liệu gốc Hyndman, Cawley–Talbot, ECB và SEC; phân biệt cơ sở phương pháp với ngày cắt thiết kế chưa tối ưu. Nhãn HMM chính đã hiển thị trên dashboard; không áp nhãn sang mô hình khác. Giữ nguyên mọi hash khoa học; kiểm tra 16 tests, lint, UI thực, download và rebuild không raw/cache. Không tạo PDF hay ước lượng lại. Lịch Windows 18:00 đã chạy nhưng exit1 do mã hóa script, sửa parser không đồng nghĩa đã xác nhận cơ chế continuation; người dùng yêu cầu tiếp tục trực tiếp.
