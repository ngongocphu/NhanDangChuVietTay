#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script để train CRNN model từ đầu cho Vietnamese Handwriting Recognition
- Báo cáo tiến độ mỗi 10 epoch
- Train liên tục đến khi hoàn thành
"""

import os
import json
import cv2
import numpy as np
import pathlib
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from multiprocessing import Pool, cpu_count
from functools import partial

# Cấu hình để tất cả file temp lưu trong dự án (ổ D)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_TEMP = os.path.join(PROJECT_ROOT, '.temp')
os.makedirs(PROJECT_TEMP, exist_ok=True)

# Set environment variables để TensorFlow dùng temp trong dự án
os.environ['TMPDIR'] = PROJECT_TEMP
os.environ['TMP'] = PROJECT_TEMP
os.environ['TEMP'] = PROJECT_TEMP
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Giảm log TensorFlow

import tensorflow as tf

# Cấu hình TensorFlow để dùng temp trong dự án
tf.config.experimental.set_memory_growth = lambda x, y: None  # Disable memory growth warning
from tensorflow.keras.layers import (
    Dense, LSTM, BatchNormalization, Input, Conv2D, 
    MaxPool2D, Lambda, Bidirectional, Add, Activation
)
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import (
    TensorBoard, ModelCheckpoint, EarlyStopping, 
    ReduceLROnPlateau, CSVLogger, Callback
)
from tensorflow.keras.optimizers import Adam

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_DIR = 'data/vn_handwritten_images'
LABELS_JSON = os.path.join(DATA_DIR, 'labels.json')
TIME_STEPS = 240
TARGET_HEIGHT = 118
TARGET_WIDTH = 2167
BATCH_SIZE = 256   # Batch size (tối ưu để đạt 12 phút/epoch cho Phase 1)
EPOCHS_PHASE1 = 50    # Phase 1: Training chính
EPOCHS_PHASE2 = 30    # Phase 2: Fine-tuning
USE_TF_DATA = True  # Sử dụng tf.data để tăng tốc
NUM_PARALLEL_CALLS = 8  # Số thread xử lý song song (tăng để tăng tốc)
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Output paths - lưu vào thư mục model (ổ D)
MODEL_DIR = os.path.join(PROJECT_ROOT, 'model')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(MODEL_DIR, 'model_checkpoint_weights.weights.h5')
CHAR_LIST_PATH = os.path.join(MODEL_DIR, 'char_list.json')
TRAINING_REPORT_FILE = os.path.join(PROJECT_ROOT, 'training_report.txt')


# ============================================================================
# CUSTOM CALLBACK - BÁO CÁO MỖI 10 EPOCH
# ============================================================================
class ProgressReportCallback(Callback):
    """Callback để báo cáo tiến độ mỗi 10 epoch"""
    
    def __init__(self, report_interval=10):
        super().__init__()
        self.report_interval = report_interval
        self.epoch_times = []
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.report_file = TRAINING_REPORT_FILE
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_epoch = epoch + 1
        
        # Tính thời gian epoch
        if hasattr(self, 'epoch_start_time'):
            epoch_time = time.time() - self.epoch_start_time
            self.epoch_times.append(epoch_time)
        
        # Cập nhật best model
        val_loss = logs.get('val_loss', float('inf'))
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = current_epoch
        
        # Báo cáo mỗi 10 epoch
        if current_epoch % self.report_interval == 0:
            self._print_report(current_epoch, logs)
            self._save_report(current_epoch, logs)
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
    
    def _print_report(self, epoch, logs):
        """In báo cáo ra console"""
        print("\n" + "=" * 70)
        print(f"📊 BÁO CÁO TIẾN ĐỘ - EPOCH {epoch}")
        print("=" * 70)
        
        train_loss = logs.get('loss', 'N/A')
        val_loss = logs.get('val_loss', 'N/A')
        lr = float(K.get_value(self.model.optimizer.learning_rate))
        
        print(f"📈 Loss:")
        print(f"   Training Loss:   {train_loss:.4f}")
        print(f"   Validation Loss: {val_loss:.4f}")
        print(f"   Learning Rate:   {lr:.6f}")
        
        # Thống kê thời gian
        if len(self.epoch_times) >= self.report_interval:
            avg_time = np.mean(self.epoch_times[-self.report_interval:])
            total_time = sum(self.epoch_times)
            remaining_epochs = EPOCHS - epoch
            est_remaining = avg_time * remaining_epochs
            
            print(f"\n⏰ Thời gian:")
            print(f"   Trung bình/epoch: {avg_time:.1f}s")
            print(f"   Tổng thời gian:   {total_time/60:.1f} phút")
            print(f"   Dự kiến còn lại:  {est_remaining/60:.1f} phút")
        
        print(f"\n🏆 Best Model:")
        print(f"   Best Epoch:      {self.best_epoch}")
        print(f"   Best Val Loss:   {self.best_val_loss:.4f}")
        
        # Đánh giá tình trạng học
        improvement = self.best_val_loss - val_loss
        if improvement > 0:
            print(f"\n✅ Model đang học tốt!")
            print(f"   Cải thiện: {improvement:.4f} so với best")
        elif val_loss < train_loss * 1.2:
            print(f"\n✅ Model học ổn định (không overfitting)")
        else:
            print(f"\n⚠️  Cảnh báo: Validation loss cao hơn training loss")
            print(f"   Có thể đang overfitting")
        
        print("=" * 70 + "\n")
    
    def _save_report(self, epoch, logs):
        """Lưu báo cáo vào file"""
        report = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'train_loss': float(logs.get('loss', 0)),
            'val_loss': float(logs.get('val_loss', 0)),
            'learning_rate': float(K.get_value(self.model.optimizer.learning_rate)),
            'best_epoch': self.best_epoch,
            'best_val_loss': float(self.best_val_loss)
        }
        
        # Đọc reports hiện có
        reports = []
        if os.path.exists(self.report_file):
            try:
                with open(self.report_file, 'r', encoding='utf-8') as f:
                    reports = json.load(f)
            except:
                reports = []
        
        # Thêm report mới
        reports.append(report)
        
        # Lưu lại
        with open(self.report_file, 'w', encoding='utf-8') as f:
            json.dump(reports, f, indent=2, ensure_ascii=False)

# ============================================================================
# DATA PREPROCESSING FUNCTIONS
# ============================================================================
def encode_to_labels(txt, char_list):
    """Convert text to array of character indices"""
    dig_lst = []
    for char in txt:
        try:
            dig_lst.append(char_list.index(char))
        except ValueError:
            pass  # Bỏ qua ký tự không tìm thấy để tăng tốc
    return dig_lst

def preprocess_image(img_path, target_height=TARGET_HEIGHT, target_width=TARGET_WIDTH):
    """Preprocess image for CRNN model - Tối ưu tối đa"""
    # Read and convert to grayscale - tối ưu nhất
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh: {img_path}")
    
    height, width = img.shape
    
    # Resize - sử dụng INTER_AREA cho downscale (nhanh hơn)
    if height != target_height:
        new_width = int(target_height/height*width)
        if new_width > width:
            img = cv2.resize(img, (new_width, target_height), interpolation=cv2.INTER_LINEAR)
        else:
            img = cv2.resize(img, (new_width, target_height), interpolation=cv2.INTER_AREA)
    else:
        img = img.copy()
    
    height, width = img.shape
    
    # Pad/crop to target width - tối ưu
    if width < target_width:
        # Pad với constant (nhanh hơn median)
        img = np.pad(img, ((0, 0), (0, target_width - width)), mode='constant', constant_values=128)
    elif width > target_width:
        start = (width - target_width) // 2
        img = img[:, start:start + target_width]
    
    # Giảm preprocessing để tăng tốc - chỉ giữ những gì cần thiết
    # Bỏ Gaussian blur (tiết kiệm thời gian)
    # Bỏ adaptive threshold (tiết kiệm thời gian)
    # Chỉ normalize
    
    # Add channel dimension and normalize
    img = np.expand_dims(img, axis=2)
    img = img.astype(np.float32) / 255.0
    
    return img

# ============================================================================
# TF.DATA DATASET FUNCTIONS - Tối ưu hóa tốc độ
# ============================================================================
def preprocess_image_tf(img_path_bytes, label_bytes, char_list_bytes):
    """Preprocess image using TensorFlow operations - nhanh hơn"""
    import tensorflow as tf
    
    # Decode paths
    img_path = img_path_bytes.numpy().decode('utf-8')
    label = label_bytes.numpy().decode('utf-8')
    char_list = json.loads(char_list_bytes.numpy().decode('utf-8'))
    
    # Preprocess image
    img = preprocess_image(img_path)
    
    # Encode label
    label_encoded = encode_to_labels(label, char_list)
    
    return img, label_encoded, len(label)

def create_tf_dataset(image_paths, labels, char_list, batch_size, shuffle=True, cache=True):
    """Tạo tf.data.Dataset để tăng tốc training"""
    char_list_json = json.dumps(char_list)
    
    # Tạo dataset từ paths và labels
    dataset = tf.data.Dataset.from_tensor_slices((
        [str(p) for p in image_paths],
        labels,
        [char_list_json] * len(image_paths)
    ))
    
    # Shuffle
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths), reshuffle_each_iteration=True)
    
    # Map preprocessing - song song hóa
    def py_preprocess(img_path, label, char_list_str):
        img, label_enc, label_len = tf.py_function(
            lambda x, y, z: preprocess_image_tf(x, y, z),
            [img_path, label, char_list_str],
            [tf.float32, tf.int64, tf.int64]
        )
        img.set_shape((TARGET_HEIGHT, TARGET_WIDTH, 1))
        return img, label_enc, label_len
    
    dataset = dataset.map(
        py_preprocess,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    )
    
    # Cache để tránh preprocessing lại
    if cache:
        dataset = dataset.cache()
    
    # Batch
    dataset = dataset.batch(batch_size, drop_remainder=False)
    
    # Prefetch để tăng tốc
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def prepare_tf_data(image_paths, labels, char_list, batch_size):
    """Chuẩn bị dữ liệu cho training với tf.data"""
    # Pad labels
    max_label_len = TIME_STEPS
    padded_labels = []
    label_lengths = []
    input_lengths = []
    
    for label in labels:
        encoded = encode_to_labels(label, char_list)
        padded = encoded + [0] * (max_label_len - len(encoded))
        padded = padded[:max_label_len]  # Truncate nếu quá dài
        padded_labels.append(padded)
        label_lengths.append(len(encoded))
        input_lengths.append(TIME_STEPS)
    
    # Tạo dataset
    dataset = tf.data.Dataset.from_tensor_slices((
        [str(p) for p in image_paths],
        labels,
        [json.dumps(char_list)] * len(image_paths)
    ))
    
    # Map preprocessing
    def py_preprocess(img_path, label, char_list_str):
        img, label_enc, label_len = tf.py_function(
            lambda x, y, z: preprocess_image_tf(x, y, z),
            [img_path, label, char_list_str],
            [tf.float32, tf.int64, tf.int64]
        )
        img.set_shape((TARGET_HEIGHT, TARGET_WIDTH, 1))
        return img, label_enc, label_len
    
    dataset = dataset.map(
        py_preprocess,
        num_parallel_calls=NUM_PARALLEL_CALLS,
        deterministic=False
    )
    
    # Shuffle và batch
    dataset = dataset.shuffle(buffer_size=min(1000, len(image_paths)), reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    # Convert labels to arrays
    padded_labels = np.array(padded_labels, dtype=np.int32)
    label_lengths = np.array(label_lengths, dtype=np.int32)
    input_lengths = np.array(input_lengths, dtype=np.int32)
    
    return dataset, padded_labels, label_lengths, input_lengths

# ============================================================================
# BUILD CRNN MODEL
# ============================================================================
def build_crnn_model(char_list):
    """Xây dựng CRNN model architecture"""
    print("\n" + "=" * 70)
    print("XÂY DỰNG CRNN MODEL")
    print("=" * 70)
    
    # Input layer
    inputs = Input(shape=(TARGET_HEIGHT, TARGET_WIDTH, 1))
    
    # Block 1
    x = Conv2D(64, (3, 3), padding='same')(inputs)
    x = MaxPool2D(pool_size=3, strides=3)(x)
    x = Activation('relu')(x)
    x_1 = x
    
    # Block 2
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = MaxPool2D(pool_size=3, strides=3)(x)
    x = Activation('relu')(x)
    x_2 = x
    
    # Block 3
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x_3 = x
    
    # Block 4
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, x_3])
    x = Activation('relu')(x)
    x_4 = x
    
    # Block 5
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x_5 = x
    
    # Block 6
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, x_5])
    x = Activation('relu')(x)
    
    # Block 7
    x = Conv2D(1024, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(3, 1))(x)
    x = Activation('relu')(x)
    
    # Final pooling
    x = MaxPool2D(pool_size=(3, 1))(x)
    
    # Squeeze dimension
    squeezed = Lambda(lambda x: K.squeeze(x, 1))(x)
    
    # Bidirectional LSTM layers với dropout để tránh overfitting
    blstm_1 = Bidirectional(
        LSTM(512, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)
    )(squeezed)
    blstm_2 = Bidirectional(
        LSTM(512, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)
    )(blstm_1)
    
    # Output layer
    num_classes = len(char_list) + 1  # +1 for blank token
    outputs = Dense(num_classes, activation='softmax', dtype='float32')(blstm_2)
    
    # Model for prediction (without CTC)
    act_model = Model(inputs, outputs)
    
    print(f"✅ Model architecture đã được xây dựng")
    print(f"   Số lớp: {len(act_model.layers)}")
    print(f"   Số ký tự: {num_classes}")
    
    return act_model, num_classes

def build_training_model(act_model, num_classes, learning_rate=0.0005):
    """Xây dựng model cho training với CTC loss"""
    # CTC loss function
    def ctc_lambda_func(args):
        y_pred, labels, input_length, label_length = args
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
    
    # Get inputs and outputs from act_model
    inputs = act_model.input
    outputs = act_model.output
    
    # CTC inputs
    labels_input = Input(name='the_labels', shape=[TIME_STEPS], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    
    # CTC loss layer
    loss_out = Lambda(
        ctc_lambda_func, 
        output_shape=(1,), 
        name='ctc'
    )([outputs, labels_input, input_length, label_length])
    
    # Model for training (with CTC)
    model = Model(
        inputs=[inputs, labels_input, input_length, label_length], 
        outputs=loss_out
    )
    
    # Compile model với learning rate được chỉ định
    optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
    
    model.compile(
        loss={'ctc': lambda y_true, y_pred: y_pred}, 
        optimizer=optimizer
    )
    
    print(f"✅ Training model đã được compile")
    print(f"   Tổng số tham số: {model.count_params():,}")
    print(f"   Learning rate: {learning_rate}")
    
    return model

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================
def main():
    """Hàm chính để train model"""
    print("\n" + "=" * 70)
    print("VIETNAMESE HANDWRITING RECOGNITION - CRNN TRAINING")
    print("=" * 70)
    print(f"Thời gian bắt đầu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Setup training
    batch_size = BATCH_SIZE
    
    # 2. Load labels và tạo char_list
    print("\n" + "=" * 70)
    print("LOAD DỮ LIỆU")
    print("=" * 70)
    
    if not os.path.exists(LABELS_JSON):
        raise FileNotFoundError(f"Không tìm thấy file labels: {LABELS_JSON}")
    
    print(f"Đang tải labels từ {LABELS_JSON}...")
    with open(LABELS_JSON, 'r', encoding='utf8') as f:
        labels = json.load(f)
    print(f"✅ Đã tải {len(labels)} labels")
    
    # Tạo char_list
    print("\nĐang tạo danh sách ký tự...")
    char_list = set()
    for label in labels.values():
        char_list.update(set(label))
    char_list = sorted(char_list)
    print(f"✅ Tìm thấy {len(char_list)} ký tự duy nhất")
    
    # Lưu char_list
    with open(CHAR_LIST_PATH, 'w', encoding='utf-8') as f:
        json.dump(char_list, f, ensure_ascii=False, indent=2)
    print(f"✅ Đã lưu char_list vào {CHAR_LIST_PATH}")
    
    # 3. Chuẩn bị dữ liệu
    print("\n" + "=" * 70)
    print("CHUẨN BỊ DỮ LIỆU")
    print("=" * 70)
    
    # Tạo mapping từ filepath đến label
    print("Đang tạo mapping ảnh - labels...")
    dict_filepath_label = {}
    raw_data_path = pathlib.Path(DATA_DIR)
    
    for item in raw_data_path.glob('**/*.*'):
        file_name = str(os.path.basename(item))
        if file_name != "labels.json" and file_name in labels:
            label = labels[file_name]
            dict_filepath_label[str(item)] = label
    
    print(f"✅ Tìm thấy {len(dict_filepath_label)} ảnh có labels")
    
    # Split train/validation
    all_image_paths = list(dict_filepath_label.keys())
    train_image_paths, val_image_paths = train_test_split(
        all_image_paths, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE
    )
    
    print(f"Training: {len(train_image_paths)} ảnh")
    print(f"Validation: {len(val_image_paths)} ảnh")
    
    # Xử lý ảnh training - song song hóa
    print("\nĐang xử lý ảnh training (song song hóa)...")
    num_workers = min(cpu_count(), 8)  # Tối đa 8 workers
    print(f"   Sử dụng {num_workers} workers để xử lý song song")
    
    # Xử lý tuần tự (tránh lỗi pickle trên Windows)
    print("   ⚠️  Windows: Sử dụng xử lý tuần tự để tránh lỗi multiprocessing")
    training_img = []
    training_txt = []
    train_input_length = []
    train_label_length = []
    
    for i, train_img_path in enumerate(train_image_paths):
        try:
            img = preprocess_image(train_img_path)
            label = dict_filepath_label[train_img_path]
            
            train_label_length.append(len(label))
            train_input_length.append(TIME_STEPS)
            training_img.append(img)
            training_txt.append(encode_to_labels(label, char_list))
            
            if (i + 1) % 100 == 0:
                print(f"   Đã xử lý {i + 1}/{len(train_image_paths)} ảnh")
        except Exception as e:
            print(f"⚠️  Lỗi xử lý {train_img_path}: {e}")
            continue
    
    print(f"✅ Đã xử lý {len(training_img)} ảnh training")
    
    # Xử lý ảnh validation - tuần tự
    print("\nĐang xử lý ảnh validation...")
    
    valid_img = []
    valid_txt = []
    valid_input_length = []
    valid_label_length = []
    
    for i, val_img_path in enumerate(val_image_paths):
        try:
            img = preprocess_image(val_img_path)
            label = dict_filepath_label[val_img_path]
            
            valid_label_length.append(len(label))
            valid_input_length.append(TIME_STEPS)
            valid_img.append(img)
            valid_txt.append(encode_to_labels(label, char_list))
            
            if (i + 1) % 100 == 0:
                print(f"   Đã xử lý {i + 1}/{len(val_image_paths)} ảnh")
        except Exception as e:
            print(f"⚠️  Lỗi xử lý {val_img_path}: {e}")
            continue
    
    print(f"✅ Đã xử lý {len(valid_img)} ảnh validation")
    
    # Convert to numpy arrays và pad sequences
    print("\nĐang chuẩn bị dữ liệu cuối cùng...")
    
    training_img = np.array(training_img)
    train_input_length = np.array(train_input_length)
    train_label_length = np.array(train_label_length)
    
    valid_img = np.array(valid_img)
    valid_input_length = np.array(valid_input_length)
    valid_label_length = np.array(valid_label_length)
    
    # Pad sequences
    max_label_len = TIME_STEPS
    train_padded_txt = pad_sequences(
        training_txt, 
        maxlen=max_label_len, 
        padding='post', 
        value=0
    )
    valid_padded_txt = pad_sequences(
        valid_txt, 
        maxlen=max_label_len, 
        padding='post', 
        value=0
    )
    
    print(f"✅ Kích thước dữ liệu:")
    print(f"   Training images: {training_img.shape}")
    print(f"   Validation images: {valid_img.shape}")
    print(f"   Training labels: {train_padded_txt.shape}")
    print(f"   Validation labels: {valid_padded_txt.shape}")
    
    # 4. Xây dựng model
    act_model, num_classes = build_crnn_model(char_list)
    initial_lr = 0.0005  # Learning rate cho phase 1
    fine_tune_lr = 0.0001  # Learning rate cho phase 2 (fine-tuning)
    model = build_training_model(act_model, num_classes, learning_rate=initial_lr)
    
    # 5. Tạo callbacks
    print("\n" + "=" * 70)
    print("THIẾT LẬP CALLBACKS")
    print("=" * 70)
    
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    callbacks = [
        # Custom callback để báo cáo mỗi 10 epoch
        ProgressReportCallback(report_interval=10),
        
        # TensorBoard - tối ưu tối đa để giảm overhead
        TensorBoard(
            log_dir=LOGS_DIR,
            histogram_freq=0,  # Tắt histogram để tăng tốc tối đa
            profile_batch=0,
            write_graph=False,  # Tắt write graph để tăng tốc
            write_images=False,
            update_freq=10  # Chỉ update mỗi 10 epoch để tăng tốc
        ),
        
        # ModelCheckpoint - lưu model tốt nhất (chỉ lưu khi cải thiện)
        ModelCheckpoint(
            filepath=CHECKPOINT_PATH,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=0,  # Giảm output để tăng tốc
            mode='min',
            save_freq='epoch'  # Chỉ lưu mỗi epoch
        ),
        
        # EarlyStopping - dừng nếu không cải thiện (chỉ cho phase 1)
        EarlyStopping(
            monitor='val_loss',
            min_delta=0.5,
            patience=15,  # Patience cho phase 1
            restore_best_weights=True,
            verbose=0,  # Giảm output
            mode='min'
        ),
        
        # ReduceLROnPlateau - giảm learning rate khi không cải thiện
        ReduceLROnPlateau(
            monitor='val_loss',
            min_delta=0.5,
            factor=0.5,
            patience=10,
            verbose=0,  # Giảm output
            min_lr=1e-6,
            mode='min'
        ),
        
        # CSVLogger - lưu logs vào CSV
        CSVLogger(
            filename=os.path.join(LOGS_DIR, 'training_log.csv'),
            append=False
        )
    ]
    
    print("✅ Đã tạo các callbacks:")
    print(f"   - ProgressReportCallback: Báo cáo mỗi 10 epoch")
    print(f"   - ModelCheckpoint: {CHECKPOINT_PATH}")
    print(f"   - TensorBoard: {LOGS_DIR}")
    print(f"   - EarlyStopping: patience=20")
    print(f"   - ReduceLROnPlateau: factor=0.5, patience=10")
    
    # 6. Bắt đầu training
    print("\n" + "=" * 70)
    print("BẮT ĐẦU TRAINING")
    print("=" * 70)
    print(f"Batch size: {batch_size}")
    print(f"Phase 1: {EPOCHS_PHASE1} epochs (Training chính)")
    print(f"Phase 2: {EPOCHS_PHASE2} epochs (Fine-tuning)")
    print(f"Training samples: {len(training_img)}")
    print(f"Validation samples: {len(valid_img)}")
    print("=" * 70)
    
    start_time = time.time()
    
    try:
        # ========================================================================
        # PHASE 1: TRAINING CHÍNH (50 epochs)
        # ========================================================================
        print("\n" + "=" * 70)
        print("PHASE 1: TRAINING CHÍNH")
        print("=" * 70)
        print(f"Epochs: {EPOCHS_PHASE1}")
        print(f"Learning rate: {initial_lr}")
        print("=" * 70)
        
        history_phase1 = model.fit(
            x=[
                training_img,
                train_padded_txt,
                train_input_length,
                train_label_length
            ],
            y=np.zeros(len(training_img)),
            batch_size=batch_size,
            epochs=EPOCHS_PHASE1,
            validation_data=([
                valid_img,
                valid_padded_txt,
                valid_input_length,
                valid_label_length
            ], [np.zeros(len(valid_img))]),
            verbose=1,
            callbacks=callbacks,
            validation_freq=10  # Validate mỗi 10 epoch để tăng tốc (giảm từ 5)
        )
        
        phase1_time = time.time() - start_time
        print("\n" + "=" * 70)
        print("✅ PHASE 1 HOÀN TẤT!")
        print("=" * 70)
        print(f"⏰ Thời gian Phase 1: {int(phase1_time//3600)}h {int((phase1_time%3600)//60)}m {int(phase1_time%60)}s")
        
        # ========================================================================
        # PHASE 2: FINE-TUNING (30 epochs)
        # ========================================================================
        print("\n" + "=" * 70)
        print("PHASE 2: FINE-TUNING")
        print("=" * 70)
        print(f"Epochs: {EPOCHS_PHASE2}")
        print(f"Learning rate: {fine_tune_lr} (giảm từ {initial_lr})")
        print("=" * 70)
        
        # Giảm learning rate cho fine-tuning
        K.set_value(model.optimizer.learning_rate, fine_tune_lr)
        print(f"✅ Đã giảm learning rate xuống {fine_tune_lr}")
        
        # Tạo callbacks mới cho phase 2 (không có EarlyStopping, chỉ có ModelCheckpoint)
        callbacks_phase2 = [
            ProgressReportCallback(report_interval=10),
            TensorBoard(
                log_dir=LOGS_DIR,
                histogram_freq=0,  # Tắt histogram để tăng tốc
                profile_batch=0,
                write_graph=False,  # Tắt để tăng tốc
                write_images=False,
                update_freq="epoch"
            ),
            ModelCheckpoint(
                filepath=CHECKPOINT_PATH,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True,
                verbose=0,  # Giảm output
                mode='min',
                save_freq='epoch'
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                min_delta=0.3,
                factor=0.3,  # Giảm LR mạnh hơn trong fine-tuning
                patience=8,
                verbose=0,  # Giảm output
                min_lr=1e-7,
                mode='min'
            ),
            CSVLogger(
                filename=os.path.join(LOGS_DIR, 'training_log.csv'),
                append=True  # Append vào log của phase 1
            )
        ]
        
        history_phase2 = model.fit(
            x=[
                training_img,
                train_padded_txt,
                train_input_length,
                train_label_length
            ],
            y=np.zeros(len(training_img)),
            batch_size=batch_size,
            epochs=EPOCHS_PHASE1 + EPOCHS_PHASE2,  # Tổng số epochs
            initial_epoch=EPOCHS_PHASE1,  # Tiếp tục từ epoch 50
            validation_data=([
                valid_img,
                valid_padded_txt,
                valid_input_length,
                valid_label_length
            ], [np.zeros(len(valid_img))]),
            verbose=1,
            callbacks=callbacks_phase2,
            validation_freq=10  # Validate mỗi 10 epoch để tăng tốc (giảm từ 5)
        )
        
        elapsed_time = time.time() - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        
        print("\n" + "=" * 70)
        print("✅ TRAINING HOÀN TẤT!")
        print("=" * 70)
        print(f"⏰ Tổng thời gian: {hours}h {minutes}m {seconds}s")
        print(f"   - Phase 1: {EPOCHS_PHASE1} epochs")
        print(f"   - Phase 2: {EPOCHS_PHASE2} epochs (fine-tuning)")
        print(f"📦 Model: {CHECKPOINT_PATH}")
        print(f"📝 Char list: {CHAR_LIST_PATH}")
        print(f"📊 Logs: {LOGS_DIR}")
        print(f"📄 Training report: {TRAINING_REPORT_FILE}")
        print(f"✅ Model đã được lưu vào thư mục model/")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training bị dừng bởi người dùng")
        print(f"Model đã được lưu tại: {CHECKPOINT_PATH}")
    except Exception as e:
        print(f"\n\n❌ Lỗi trong quá trình training: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()

