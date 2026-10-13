# F5-TTS Voice Clone Service

Microservice Python độc lập dùng F5-TTS để tổng hợp tiếng nói zero-shot từ một file WAV mẫu và transcript tương ứng. Model được giữ trong bộ nhớ đúng một lần cho mỗi process. Backend được chọn theo thứ tự **MPS → CUDA → CPU**.

> Chỉ sử dụng giọng nói khi bạn có sự đồng ý rõ ràng của chủ giọng. Không dùng dịch vụ này để mạo danh, lừa đảo hoặc tạo nội dung gây hiểu nhầm. Mã F5-TTS là MIT, nhưng checkpoint pretrained có điều khoản CC-BY-NC; hãy kiểm tra giấy phép trước khi triển khai thương mại.

## Cấu trúc

```text
voice-clone-service/
├── voices/                 # Các cặp <voice_name>.wav + <voice_name>.txt
├── outputs/                # WAV đã render
├── src/
│   ├── __init__.py
│   ├── engine.py
│   └── text_splitter.py
├── app.py
├── cli.py
├── requirements.txt
└── README.md
```

Repository không đóng gói giọng thật hoặc giọng giả mẫu. Trước khi chạy, hãy tự thêm profile đã được cấp phép như hướng dẫn bên dưới.

## Yêu cầu hệ thống

- Python 3.10 hoặc 3.11 (khuyến nghị 3.11)
- Khoảng 8 GB RAM trở lên; MPS/CUDA được khuyến nghị
- Kết nối Internet trong lần chạy đầu để F5-TTS tải checkpoint mặc định
- File WAV tham chiếu sạch, một người nói, ít tạp âm

## Cài đặt

### Apple Silicon (M1/M2/M3/M4...)

```bash
cd voice-clone-service
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch chính thức cho macOS đã bao gồm MPS. Có thể xác nhận bằng:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

### NVIDIA CUDA

Cài wheel PyTorch phù hợp với driver/CUDA trước, ví dụ CUDA 12.8, rồi cài phần còn lại:

```bash
cd voice-clone-service
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Kiểm tra GPU:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

### CPU

Dùng quy trình cài đặt Apple Silicon ở trên. Engine tự rơi về CPU nếu MPS và CUDA đều không sẵn sàng. CPU chậm hơn đáng kể.

## Thêm voice profile

Mỗi profile cần hai file cùng tên:

```text
voices/deep_male.wav
voices/deep_male.txt
```

- `deep_male.wav`: đoạn nói tham chiếu rõ ràng, nên dài khoảng 5–12 giây.
- `deep_male.txt`: transcript UTF-8 chính xác của toàn bộ nội dung trong WAV, chỉ gồm văn bản thực sự được nói.
- Tên profile chỉ dùng chữ Latin, số, `_` và `-`; tối đa 64 ký tự.

Ví dụ nội dung `voices/deep_male.txt`:

```text
Selamat datang, hari ini kita akan membahas teknologi suara dengan jelas dan tenang.
```

Chất lượng đầu ra phụ thuộc mạnh vào độ sạch của audio, độ chính xác transcript và khả năng ngôn ngữ của checkpoint đang dùng.

## Chạy REST API

```bash
source .venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 8000
```

Model được preload trong lifecycle khởi động. Lần đầu có thể mất thời gian vì checkpoint được tải và nạp vào RAM/VRAM. Không dùng nhiều worker trên một GPU nếu không muốn mỗi worker nạp một bản model riêng.

Swagger UI: <http://localhost:8000/docs>

### Health check

```bash
curl http://localhost:8000/health
```

Ví dụ phản hồi:

```json
{
  "status": "ok",
  "device": "mps",
  "model_loaded": true
}
```

### Liệt kê voice

```bash
curl http://localhost:8000/voices
```

```json
{
  "voices": ["deep_male"],
  "count": 1
}
```

Chỉ profile có đủ cả `.wav` và `.txt` mới được liệt kê.

### Tổng hợp và nhận metadata JSON

```bash
curl -X POST "http://localhost:8000/v1/tts/synthesize" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Selamat datang. Hari ini kita akan mencoba sistem sintesis suara yang baru.",
    "voice_name": "deep_male",
    "speed": 0.9,
    "nfe_step": 32
  }'
```

```json
{
  "status": "success",
  "output_path": "/absolute/path/to/voice-clone-service/outputs/20261013-120000-<uuid>.wav",
  "duration_seconds": 4.582
}
```

### Stream WAV trực tiếp

```bash
curl -X POST "http://localhost:8000/v1/tts/synthesize?stream=true" \
  -H "Content-Type: application/json" \
  -d '{"text":"Selamat pagi, semoga hari Anda menyenangkan.","voice_name":"deep_male"}' \
  --output result.wav
```

## Chạy CLI

```bash
source .venv/bin/activate
python cli.py \
  --text "Selamat datang di layanan sintesis suara." \
  --voice deep_male \
  --speed 0.9 \
  --nfe-step 32 \
  --output test.wav
```

Nếu bỏ `--output`, file được giữ trong `outputs/` với tên timestamp và UUID.

## Tham số

| Tham số | Mặc định | Giới hạn API | Ý nghĩa |
|---|---:|---:|---|
| `text` | bắt buộc | 1–20.000 ký tự | Văn bản cần đọc |
| `voice_name` | `deep_male` | 1–64 ký tự | Tên cặp WAV/TXT |
| `speed` | `0.9` | `> 0.1`, `≤ 2.0` | Tốc độ nói |
| `nfe_step` | `32` | 1–128 | Số bước flow matching; cao hơn thường chậm hơn |
| `stream` | `false` | boolean query | Trả WAV thay vì JSON |

Văn bản dài được chia thành các đoạn ưu tiên ranh giới câu, mục tiêu 20–40 từ. Engine render tuần tự và chèn 250 ms im lặng giữa các đoạn trước khi ghi một WAV PCM 16-bit duy nhất.

## Biến môi trường

| Biến | Mặc định | Mô tả |
|---|---|---|
| `VOICES_DIR` | `<project>/voices` | Thư mục profile giọng |
| `OUTPUTS_DIR` | `<project>/outputs` | Thư mục WAV kết quả |
| `F5_MODEL_NAME` | `F5TTS_v1_Base` | Tên config model F5-TTS |
| `F5_CHECKPOINT_FILE` | trống | Checkpoint cục bộ tùy chọn |
| `F5_VOCAB_FILE` | trống | Vocabulary cục bộ tùy chọn |

Ví dụ:

```bash
F5_CHECKPOINT_FILE=/models/model_1250000.safetensors \
F5_VOCAB_FILE=/models/vocab.txt \
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Mã lỗi HTTP

- `404`: thiếu WAV hoặc TXT của voice profile.
- `422`: JSON/tham số không hợp lệ, tên voice không an toàn, audio lỗi hoặc transcript rỗng.
- `500`: F5-TTS inference thất bại; chi tiết kỹ thuật chỉ ghi ở server log.
- `503`: hết RAM/VRAM trên MPS/CUDA/CPU.

Log ghi backend, số đoạn, thời gian load model, thời gian inference và thời lượng audio. Các request synthesis được tuần tự hóa trong một process để tránh model bị gọi đồng thời và tăng đột biến VRAM.

## Nguồn kỹ thuật

- [F5-TTS chính thức](https://github.com/SWivid/F5-TTS)
- [F5-TTS 1.1.22 trên PyPI](https://pypi.org/project/f5-tts/1.1.22/)
- [PyTorch MPS backend](https://pytorch.org/docs/stable/notes/mps.html)

Nội dung tham khảo được diễn giải lại để tuân thủ giới hạn giấy phép nội dung.
