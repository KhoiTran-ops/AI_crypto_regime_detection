# Audit tái lập phiên bản v1

Lượt tái tính dùng raw cache và cùng môi trường khóa, xuất sang TemporaryDirectory riêng. Toàn bộ 18 mô hình được fit lại. Các kết quả HMM/GMM/K-means khớp, nhưng các mô hình econometric khác ở chữ số nhỏ; ví dụ mean log score BTC NHHMM từ 2,4210047426003007 thành 2,420988950012428. Kiểm tra rtol=1e-8 / atol=1e-10 dừng; không nới dung sai để qua.

Nguồn: statsmodels 0.15, tsa/regime_switching/markov_switching.py, fit và _start_params_search. Tài liệu ngay trong mã nói rng=None tạo Generator từ entropy hệ điều hành; np.random.seed không kiểm soát Generator này. Bản sửa truyền rng=11 trực tiếp. Kiểm thử nhỏ fit hai lần MS0 phải khớp chính xác.

Dữ liệu, số trạng thái chính BTC4/ETH5, lưới mô hình, feature, clock thực thi, chi phí và các ngưỡng precision không được chọn lại từ điểm test. Validation econometric được tính lại để đối chiếu, giữ dấu vết quyết định cũ. v1 đã xem holdout, nên bản v2 sửa lỗi không được quảng bá là holdout mới untouched.

Full v1 giữ cục bộ tại outputs/prior_runs/v1_unseeded_statsmodels (bỏ khỏi Git). Protocol và bảng v1 gọn nằm trong docs/history để người xem repository truy vết.

Ngoài RNG, v2 chuẩn hóa source/config về LF và giữ byte các artifact/snapshot bằng .gitattributes để hash không sai chỉ vì Git chuyển newline. Đây là sửa khả năng tái lập, không thay dữ liệu số.
