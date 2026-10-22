# AI Crypto Regime Detection

Yêu cầu **Python 3.12** và **Git**. Chạy các lệnh từ thư mục gốc repository.

## Cài đặt

### Windows · PowerShell

```powershell
git clone https://github.com/KhoiTran-ops/AI_crypto_regime_detection.git
cd AI_crypto_regime_detection
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
```

### Linux / macOS

```bash
git clone https://github.com/KhoiTran-ops/AI_crypto_regime_detection.git
cd AI_crypto_regime_detection
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
```

## Chạy dashboard

Windows:

```powershell
.\run_dashboard.cmd
```

Linux / macOS:

```bash
.venv/bin/python -m streamlit run dashboard.py --server.address 127.0.0.1
```

Mở **http://127.0.0.1:8501**. Dừng server bằng `Ctrl+C`.

Đổi cổng trên Windows khi cần:

```powershell
.\run_dashboard.cmd --server.port 8502
```

Dashboard có thể chạy ngay sau cài đặt bằng các file có sẵn trong repo.

## Cập nhật trực tiếp

Dashboard tự khởi động worker dùng chung: giá mỗi 60 giây; mô hình cập nhật sau khi ngày UTC hoàn tất. FRED chỉ tải một đợt mỗi ngày UTC (từ khoảng 07:05 giờ Việt Nam); nếu lỗi, chờ ngày kế tiếp, không tự thử lặp trong ngày. Giữ server chạy để duy trì cập nhật. Dữ liệu/cache live nằm trong `data/live/`, không cần commit. Năm tab hiện có tự cập nhật và giữ đầy đủ lịch sử; gói kết quả đã chốt được bảo toàn.

Chạy worker độc lập (không cần mở trình duyệt):

```powershell
.\.venv\Scripts\python.exe -m regime.cli live
```

Chạy một lượt rồi thoát:

```powershell
.\.venv\Scripts\python.exe -m regime.cli live --once
```

Dừng bằng `Ctrl+C`. Worker đang chạy sẽ được tái sử dụng, không nhân số lượt gọi theo số người xem. Khi API yêu cầu chờ, chương trình lưu cooldown và tự thử lại; không đổi IP/endpoint để né chặn.

Chỉ xem cache, không tự khởi động worker:

```powershell
$env:CRYPTO_LIVE_AUTOSTART = "0"
.\run_dashboard.cmd
```

Trên Linux/macOS, dùng `CRYPTO_LIVE_AUTOSTART=0 .venv/bin/python -m streamlit run dashboard.py`.

## Chạy pipeline

Các lệnh tải dữ liệu cần Internet, không cần API key. Chạy lần lượt:

```powershell
$env:PYTHONUTF8 = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
.\.venv\Scripts\python.exe -m regime.cli data
.\.venv\Scripts\python.exe -m regime.cli select
.\.venv\Scripts\python.exe -m regime.cli adequacy
.\.venv\Scripts\python.exe -m regime.cli evaluate
.\.venv\Scripts\python.exe -m regime.cli intervals
.\.venv\Scripts\python.exe -m regime.cli external-data
.\.venv\Scripts\python.exe -m regime.cli extend
```

`extend` tái sử dụng cache hợp lệ. Dùng `extend --recompute` để dựng lại dữ liệu/bảng với cache mô hình hiện có.

Trên Linux/macOS, dùng `.venv/bin/python` thay cho `.\.venv\Scripts\python.exe`; đặt biến môi trường bằng `export PYTHONUTF8=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`.

## Kiểm tra

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check --config pyproject.toml regime tests tools dashboard.py
.\.venv\Scripts\python.exe -m regime.cli verify-extensions
```

Xem toàn bộ tùy chọn CLI:

```powershell
.\.venv\Scripts\python.exe -m regime.cli --help
```
