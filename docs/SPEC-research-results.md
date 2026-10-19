# Spec: research-results
Chạy python -m regime.cli extend để tạo results/extensions/ độc lập baseline.
Các lựa chọn từ validation hiện tại; giai đoạn test đã quan sát nên mở rộng là hồi cứu, không nhận là untouched.
Score chung là predictive return log density (Jacobian đơn vị đúng), không so joint density khác chiều; HDBSCAN không có density.
So sánh TVTP với homogeneous cùng cohort, cùng ngày và train. Lưu hệ số, diagnostics/refit failure, ngày thiếu và source/config/code hashes.
Danh mục dùng đúng hàm/chi phí/volatility target hiện có; chỉ mô hình có p_high_vol có căn cứ từ variance mới đưa vào regime strategy. HDBSCAN noise không được tự động giảm exposure.
Không có ngưỡng RMSE mới; không chạy lại adequacy hoặc sửa dữ liệu baseline. Không tạo paper/PDF hoặc gọi publish.
