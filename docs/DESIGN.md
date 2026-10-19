# Dashboard nghiên cứu

Tham khảo: https://21st.dev/@efferd/components/efferd-dashboard-2, xem trực tiếp bằng Edge ngày 19/10/2026; Firecrawl scrape markdown/branding/screenshot cùng URL. Screenshot thực tế: sidebar sáng, nút điều hướng dọc, nền trung tính, KPI viền mảnh và lưới biểu đồ màu teal. Branding của trang 21st (General Sans, xanh #0033FF) thuộc trang chứa component, không coi là token chính xác của dashboard nhúng. Không sao chép code hoặc assets.

Lựa chọn triển khai: Streamlit hiện có, font hệ thống hỗ trợ tiếng Việt; nền #f7f8fa, card trắng, chữ #182230, chữ phụ #526175, đường viền #e4e8ef, accent teal #147d78, điểm nhấn xanh #4263eb. Bo góc 12 px, khoảng cách 16–24 px là lựa chọn của dự án. Sidebar 240px, 5 nút Tổng quan/Mô hình/Danh mục/Trạng thái/Dữ liệu; bộ chọn tài sản ở đầu nội dung. KPI 4 cột, chart/cards 2:1 hoặc 1:1 tùy nội dung; mobile dùng reflow và menu thu gọn của Streamlit.

Tất cả mô hình trong một explorer. Mật độ return và joint-core là hai mục tiêu khác nhau; clustering trình bày noise/membership/chuyển cụm. Danh mục chọn được mọi mô hình có signal; đường tài sản ghi rõ chi phí 10 bps, bảng kiểm tra cả 5/10/25 bps theo cấu hình giữ nguyên. Dữ liệu và phương pháp tập hợp một trang, không có nhật ký thực hiện. Giữ nguyên định nghĩa R và phương pháp nghiên cứu.
