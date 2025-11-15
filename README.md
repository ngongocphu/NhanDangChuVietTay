<h2 align="center">
    <a href="https://dainam.edu.vn/vi/khoa-cong-nghe-thong-tin">
    🎓 Faculty of Information Technology (DaiNam University)
    </a>
</h2>
<h2 align="center">
   XÂY DỰNG HỆ THỐNG CHUYỂN ĐỔI TÀI LIỆU VIẾT TAY THÀNH VĂN BẢN SỐ
</h2>
<div align="center">
    <p align="center">
        <img src="docs/aiotlab_logo.png" alt="AIoTLab Logo" width="170"/>
        <img src="docs/fitdnu_logo.png" alt="AIoTLab Logo" width="180"/>
        <img src="docs/dnu_logo.png" alt="DaiNam University Logo" width="200"/>
    </p>

[![AIoTLab](https://img.shields.io/badge/AIoTLab-green?style=for-the-badge)](https://www.facebook.com/DNUAIoTLab)
[![Faculty of Information Technology](https://img.shields.io/badge/Faculty%20of%20Information%20Technology-blue?style=for-the-badge)](https://dainam.edu.vn/vi/khoa-cong-nghe-thong-tin)
[![DaiNam University](https://img.shields.io/badge/DaiNam%20University-orange?style=for-the-badge)](https://dainam.edu.vn)

</div>


![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13.0-orange.svg)
![Flask](https://img.shields.io/badge/Flask-2.3.3-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)



## 🎯 Giới thiệu

Dự án này là một hệ thống OCR (Optical Character Recognition) chuyên dụng cho việc nhận dạng chữ viết tay tiếng Việt. Hệ thống sử dụng:

- **CRNN (CNN + RNN)**: Model deep learning được train từ đầu trên dataset chữ viết tay tiếng Việt
- **PaddleOCR**: Công cụ OCR mã nguồn mở của Baidu
- **EasyOCR**: Thư viện OCR đa ngôn ngữ
- **Combined OCR**: Kết hợp CRNN và PaddleOCR để tận dụng ưu điểm của cả hai

## ✨ Tính năng

- ✅ Nhận dạng chữ viết tay tiếng Việt với độ chính xác cao
- ✅ Hỗ trợ đầy đủ bảng chữ cái tiếng Việt (140+ ký tự)
- ✅ Nhiều phương thức OCR: CRNN, PaddleOCR, EasyOCR, Combined
- ✅ Giao diện web thân thiện với Flask
- ✅ API RESTful để tích hợp vào ứng dụng khác
- ✅ Batch processing cho nhiều ảnh cùng lúc
- ✅ Export kết quả ra Word (.docx) và PDF
- ✅ Training model từ đầu hoặc fine-tuning
- ✅ Hỗ trợ văn bản dài và nhiều dòng

## 💻 Yêu cầu hệ thống

- **Python**: 3.8 trở lên
- **RAM**: Tối thiểu 8GB (khuyến nghị 16GB cho training)
- **GPU**: Không bắt buộc nhưng khuyến nghị cho training (CUDA compatible)
- **Disk**: Tối thiểu 5GB trống (cho model và dependencies)

## 📦 Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/your-username/Vietnamese-Handwriting-Recognition-OCR.git
cd Vietnamese-Handwriting-Recognition-OCR
```

### 2. Tạo virtual environment (khuyến nghị)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

**Lưu ý**: 
- Cài đặt PaddlePaddle có thể mất vài phút
- Nếu gặp lỗi với TensorFlow, thử cài đặt phiên bản cụ thể:
  ```bash
  pip install tensorflow==2.13.0
  ```

### 4. Tải dataset (tùy chọn)

Dataset được cung cấp bởi Cinnamon AI. Bạn có thể tải từ:
- [Google Drive](https://drive.google.com/file/d/1-hAGX91o45NA4nv1XUYw5pMw4jMmhsh5/view?usp=sharing)
- Giải nén vào thư mục `data/vn_handwritten_images/`

## 📁 Cấu trúc dự án

```
Vietnamese-Handwriting-Recognition-OCR/
│
├── data/                          # Dữ liệu training
│   ├── vn_handwritten_images/     # Ảnh chữ viết tay
│   ├── vn_handwritten_labels/     # Labels tương ứng
│   └── README.md
│
├── model/                         # Model và cấu hình
│   ├── crnn_model.py             # Định nghĩa CRNN architecture
│   ├── char_list.json            # Danh sách ký tự (140 ký tự)
│   └── model_checkpoint_weights.weights.h5  # Model weights (sau khi train)
│
├── templates/                     # HTML templates cho web interface
│   ├── base.html
│   ├── index.html
│   ├── info.html
│   └── samples.html
│
├── logs/                          # Training logs
│
├── flask_ocr_app.py              # Flask web application
├── model_loader.py               # Load và quản lý model
├── utils.py                      # Utility functions
│
├── train_crnn_from_scratch.py    # Training CRNN từ đầu
├── train_memorize_data.py        # Training để học thuộc dữ liệu
├── train_easyocr.py              # Training EasyOCR
│
├── ocr_combined_crnn_paddle.py   # Combined OCR (CRNN + PaddleOCR)
├── ocr_without_training.py       # OCR không cần train (EasyOCR/PaddleOCR)
├── paddleocr_handwritten.py      # PaddleOCR cho chữ viết tay
├── paddleocr_long_text.py        # PaddleOCR cho văn bản dài
│
├── batch_ocr.py                  # Batch processing
├── batch_ocr_anhTT.py            # Batch processing cho thư mục cụ thể
│
├── requirements.txt              # Python dependencies
├── start_server.bat              # Script khởi động server (Windows)
├── HUONG_DAN_TRAIN.md           # Hướng dẫn training
└── README.md                     # File này
```

## 🚀 Sử dụng

### 1. Sử dụng qua Web Interface

#### Khởi động server:

**Windows:**
```bash
start_server.bat
```

**Linux/Mac:**
```bash
python flask_ocr_app.py
```

Sau đó mở trình duyệt và truy cập: `http://localhost:5000`

#### Tính năng web interface:
- Upload ảnh và nhận dạng ngay lập tức
- Xem kết quả với confidence score
- Export kết quả ra Word hoặc PDF
- Batch upload nhiều ảnh
- Xem samples và thông tin hệ thống

### 2. Sử dụng qua Python API

#### Sử dụng CRNN Model:

```python
from model_loader import ModelLoader
from utils import preprocess_image, decode_predictions

# Load model
model = ModelLoader()

# Preprocess ảnh
img = preprocess_image('path/to/image.jpg')

# Predict
predictions = model.predict(img)

# Decode kết quả
text = decode_predictions(predictions, model.char_list, greedy=True)
print(text)
```

#### Sử dụng Combined OCR (CRNN + PaddleOCR):

```python
from ocr_combined_crnn_paddle import CombinedOCR

# Khởi tạo
ocr = CombinedOCR(use_paddle=True)

# Nhận dạng
result = ocr.recognize('path/to/image.jpg', method='combined')
print(f"Text: {result['text']}")
print(f"Method: {result['method']}")  # 'crnn' hoặc 'paddle'
print(f"Confidence: {result['confidence']}%")
```

#### Sử dụng PaddleOCR:

```python
from paddleocr_handwritten import PaddleOCRHandwritten

ocr = PaddleOCRHandwritten(lang='vi')
result = ocr.recognize_with_boxes('path/to/image.jpg', return_image=False)
print(result['text'])
```

### 3. Batch Processing

```python
from batch_ocr import batch_ocr

# Xử lý nhiều ảnh trong thư mục
results = batch_ocr(
    image_dir='path/to/images',
    output_file='results.json',
    method='combined'
)
```

## 🎓 Training Model

### 1. Training CRNN từ đầu

```bash
python train_crnn_from_scratch.py
```

**Cấu hình:**
- **Phase 1**: 50 epochs với learning rate 0.0005
- **Phase 2**: 30 epochs fine-tuning với learning rate 0.0001
- **Batch size**: 256
- **Input size**: 118 x 2167 pixels
- **Time steps**: 240

**Output:**
- Model weights: `model/model_checkpoint_weights.weights.h5`
- Char list: `model/char_list.json`
- Training logs: `logs/training_log.csv`

### 2. Training để học thuộc dữ liệu

```bash
python train_memorize_data.py
```

**Cấu hình:**
- **Epochs**: 100
- **Learning rate**: 0.001
- **Batch size**: 128
- Model sẽ overfit trên training data để đạt độ chính xác cao nhất

**Output:**
- Model weights: `model/model_memorized_weights.weights.h5`

### 3. Xem hướng dẫn chi tiết

Xem file `HUONG_DAN_TRAIN.md` để biết thêm chi tiết về training.

## 🌐 Web Interface

### Routes

- **`/`**: Trang chủ - Upload và nhận dạng ảnh
- **`/info`**: Thông tin về hệ thống
- **`/samples`**: Xem các mẫu ảnh và kết quả
- **`/api/predict`**: API endpoint cho prediction (POST)
- **`/api/batch`**: API endpoint cho batch processing (POST)

### API Endpoints

#### POST `/api/predict`

Nhận dạng văn bản từ ảnh.

**Request:**
```json
{
  "image": "base64_encoded_image",
  "method": "combined"  // "crnn", "paddleocr", "easyocr", "combined"
}
```

**Response:**
```json
{
  "text": "Văn bản đã nhận dạng",
  "confidence": 95.5,
  "method": "crnn"
}
```

## 🏗️ Cấu trúc Model

### CRNN Architecture

Model CRNN bao gồm:

1. **CNN Feature Extraction** (7 blocks):
   - Conv2D layers với filters: 64 → 128 → 256 → 256 → 512 → 512 → 1024
   - MaxPooling và BatchNormalization
   - Residual connections ở block 4 và 6

2. **RNN Sequence Modeling**:
   - 2 Bidirectional LSTM layers (512 units mỗi lớp)
   - Dropout: 0.3 (training) / 0.2 (inference)

3. **Output Layer**:
   - Dense layer với softmax activation
   - 141 classes (140 ký tự + 1 blank token cho CTC)

**Tổng số tham số**: ~22 triệu (83.79 MB)

### Character Set

Model hỗ trợ 140 ký tự tiếng Việt bao gồm:
- Chữ cái in hoa và in thường (A-Z, a-z)
- Số (0-9)
- Dấu tiếng Việt đầy đủ (à, á, ả, ã, ạ, ă, â, đ, ê, ô, ơ, ư, ...)
- Ký tự đặc biệt (dấu câu, ký tự toán học)

Xem chi tiết trong `model/char_list.json`.

## 📊 Kết quả

- **Accuracy trên test set**: ~85-90% (tùy thuộc vào chất lượng ảnh)
- **Inference time**: ~0.5-1 giây/ảnh (CPU), ~0.1-0.2 giây/ảnh (GPU)
- **Model size**: ~84 MB

## 🔧 Troubleshooting

### Lỗi khi load model

```bash
# Kiểm tra file model có tồn tại không
ls model/model_checkpoint_weights.weights.h5

# Nếu không có, cần train model trước
python train_crnn_from_scratch.py
```

### Lỗi memory khi training

- Giảm `BATCH_SIZE` trong file training
- Sử dụng `USE_TF_DATA = True` để tối ưu memory
- Giảm số workers trong multiprocessing

### Lỗi PaddleOCR

```bash
# Cài đặt lại PaddleOCR
pip uninstall paddlepaddle paddleocr
pip install paddlepaddle paddleocr
```

## 🤝 Đóng góp

Mọi đóng góp đều được chào đón! Vui lòng:

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

## 📝 License

Dự án này được phân phối dưới giấy phép MIT. Xem file `LICENSE` để biết thêm chi tiết.

## 🙏 Acknowledgments

- **Dataset**: Cinnamon AI - Vietnamese Handwriting Dataset
- **PaddleOCR**: Baidu PaddlePaddle team
- **EasyOCR**: Jaided AI
- **TensorFlow**: Google

## 📧 Liên hệ

Nếu có câu hỏi hoặc gặp vấn đề, vui lòng mở issue trên GitHub.

---

**Made with ❤️ for Vietnamese OCR**

