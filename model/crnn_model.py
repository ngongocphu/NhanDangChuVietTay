#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CRNN Model Architecture cho Vietnamese Handwriting Recognition
- Định nghĩa architecture CRNN (CNN + RNN) cho nhận dạng chữ viết tay tiếng Việt
- Sử dụng CTC loss cho training
- Tương thích với train_crnn_from_scratch.py và model_loader.py
"""

import os
import json
import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, LSTM, BatchNormalization, Input, Conv2D, 
    MaxPool2D, Lambda, Bidirectional, Add, Activation
)
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


# ============================================================================
# CONFIGURATION
# ============================================================================
TARGET_HEIGHT = 118
TARGET_WIDTH = 2167
TIME_STEPS = 240


# ============================================================================
# BUILD CRNN MODEL ARCHITECTURE
# ============================================================================
def build_crnn_model(char_list, dropout_rate=0.3):
    """
    Xây dựng CRNN model architecture cho prediction
    
    Args:
        char_list: Danh sách ký tự (list of strings)
        dropout_rate: Dropout rate cho LSTM layers (default: 0.3)
    
    Returns:
        act_model: Model cho prediction (không có CTC layer)
        num_classes: Số lượng classes (số ký tự + 1 cho blank token)
    """
    # Input layer
    inputs = Input(shape=(TARGET_HEIGHT, TARGET_WIDTH, 1), name='input_image')
    
    # ========================================================================
    # CNN FEATURE EXTRACTION LAYERS
    # ========================================================================
    
    # Block 1: Conv2D 64 filters
    x = Conv2D(64, (3, 3), padding='same', name='conv2d_1')(inputs)
    x = MaxPool2D(pool_size=3, strides=3, name='maxpool2d_1')(x)
    x = Activation('relu', name='activation_1')(x)
    x_1 = x
    
    # Block 2: Conv2D 128 filters
    x = Conv2D(128, (3, 3), padding='same', name='conv2d_2')(x)
    x = MaxPool2D(pool_size=3, strides=3, name='maxpool2d_2')(x)
    x = Activation('relu', name='activation_2')(x)
    x_2 = x
    
    # Block 3: Conv2D 256 filters với BatchNormalization
    x = Conv2D(256, (3, 3), padding='same', name='conv2d_3')(x)
    x = BatchNormalization(name='batchnorm_3')(x)
    x = Activation('relu', name='activation_3')(x)
    x_3 = x  # Lưu cho residual connection
    
    # Block 4: Conv2D 256 filters với residual connection
    x = Conv2D(256, (3, 3), padding='same', name='conv2d_4')(x)
    x = BatchNormalization(name='batchnorm_4')(x)
    x = Add(name='add_1')([x, x_3])  # Residual connection
    x = Activation('relu', name='activation_4')(x)
    x_4 = x
    
    # Block 5: Conv2D 512 filters
    x = Conv2D(512, (3, 3), padding='same', name='conv2d_5')(x)
    x = BatchNormalization(name='batchnorm_5')(x)
    x = Activation('relu', name='activation_5')(x)
    x_5 = x  # Lưu cho residual connection
    
    # Block 6: Conv2D 512 filters với residual connection
    x = Conv2D(512, (3, 3), padding='same', name='conv2d_6')(x)
    x = BatchNormalization(name='batchnorm_6')(x)
    x = Add(name='add_2')([x, x_5])  # Residual connection
    x = Activation('relu', name='activation_6')(x)
    
    # Block 7: Conv2D 1024 filters
    x = Conv2D(1024, (3, 3), padding='same', name='conv2d_7')(x)
    x = BatchNormalization(name='batchnorm_7')(x)
    x = MaxPool2D(pool_size=(3, 1), name='maxpool2d_3')(x)
    x = Activation('relu', name='activation_7')(x)
    
    # Final pooling
    x = MaxPool2D(pool_size=(3, 1), name='maxpool2d_4')(x)
    
    # ========================================================================
    # SEQUENCE MODELING LAYERS (RNN)
    # ========================================================================
    
    # Squeeze dimension để chuyển từ 4D sang 3D cho LSTM
    squeezed = Lambda(lambda x: K.squeeze(x, 1), name='squeeze')(x)
    
    # Bidirectional LSTM layers
    blstm_1 = Bidirectional(
        LSTM(512, return_sequences=True, dropout=dropout_rate, recurrent_dropout=dropout_rate),
        name='bidirectional_lstm_1'
    )(squeezed)
    
    blstm_2 = Bidirectional(
        LSTM(512, return_sequences=True, dropout=dropout_rate, recurrent_dropout=dropout_rate),
        name='bidirectional_lstm_2'
    )(blstm_1)
    
    # ========================================================================
    # OUTPUT LAYER
    # ========================================================================
    
    # Output layer: Dense với softmax activation
    num_classes = len(char_list) + 1  # +1 for blank token (CTC)
    outputs = Dense(num_classes, activation='softmax', dtype='float32', name='output')(blstm_2)
    
    # Model cho prediction (không có CTC layer)
    act_model = Model(inputs=inputs, outputs=outputs, name='CRNN_Model')
    
    return act_model, num_classes


def build_training_model(act_model, num_classes, learning_rate=0.0005):
    """
    Xây dựng model cho training với CTC loss
    
    Args:
        act_model: Model architecture (từ build_crnn_model)
        num_classes: Số lượng classes
        learning_rate: Learning rate cho optimizer
    
    Returns:
        model: Model cho training (có CTC loss layer)
    """
    from tensorflow.keras.optimizers import Adam
    
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
    
    # Model cho training (với CTC)
    model = Model(
        inputs=[inputs, labels_input, input_length, label_length], 
        outputs=loss_out,
        name='CRNN_Training_Model'
    )
    
    # Compile model
    optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
    
    model.compile(
        loss={'ctc': lambda y_true, y_pred: y_pred}, 
        optimizer=optimizer
    )
    
    return model


# ============================================================================
# LOAD CHAR LIST
# ============================================================================
def load_char_list(char_list_path=None):
    """
    Load character list từ file JSON
    
    Args:
        char_list_path: Đường dẫn đến file char_list.json (None = tự động tìm)
    
    Returns:
        char_list: List of characters
    """
    if char_list_path is None:
        # Tự động tìm file char_list.json
        possible_paths = [
            'model/char_list.json',
            os.path.join(os.path.dirname(__file__), 'char_list.json'),
            'char_list.json'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                char_list_path = path
                break
    
    if char_list_path is None or not os.path.exists(char_list_path):
        raise FileNotFoundError(
            f"Không tìm thấy file char_list.json. "
            f"Vui lòng tạo file này hoặc chỉ định đường dẫn."
        )
    
    with open(char_list_path, 'r', encoding='utf-8') as f:
        char_list = json.load(f)
    
    return char_list


# ============================================================================
# MODEL SUMMARY AND INFO
# ============================================================================
def print_model_summary(model):
    """In thông tin model"""
    print("\n" + "=" * 70)
    print("CRNN MODEL SUMMARY")
    print("=" * 70)
    model.summary()
    print("=" * 70)
    print(f"Tổng số tham số: {model.count_params():,}")
    print("=" * 70 + "\n")


# ============================================================================
# MAIN - TEST FUNCTION
# ============================================================================
if __name__ == '__main__':
    """
    Test function để kiểm tra model architecture
    """
    print("=" * 70)
    print("TEST CRNN MODEL ARCHITECTURE")
    print("=" * 70)
    
    try:
        # Load char_list
        print("\n1. Đang tải char_list...")
        char_list = load_char_list()
        print(f"   ✅ Đã tải {len(char_list)} ký tự")
        
        # Build model
        print("\n2. Đang xây dựng CRNN model...")
        act_model, num_classes = build_crnn_model(char_list)
        print(f"   ✅ Model đã được xây dựng")
        print(f"   Số classes: {num_classes}")
        
        # Print summary
        print_model_summary(act_model)
        
        # Build training model
        print("\n3. Đang xây dựng training model (với CTC)...")
        training_model = build_training_model(act_model, num_classes)
        print(f"   ✅ Training model đã được compile")
        print(f"   Tổng số tham số: {training_model.count_params():,}")
        
        # Test với dummy input
        print("\n4. Test với dummy input...")
        dummy_input = tf.random.normal((1, TARGET_HEIGHT, TARGET_WIDTH, 1))
        prediction = act_model.predict(dummy_input, verbose=0)
        print(f"   ✅ Prediction shape: {prediction.shape}")
        print(f"   Expected shape: (1, {TIME_STEPS}, {num_classes})")
        
        if prediction.shape == (1, TIME_STEPS, num_classes):
            print("   ✅ Shape khớp với mong đợi!")
        else:
            print("   ⚠️  Shape không khớp!")
        
        print("\n" + "=" * 70)
        print("✅ TEST HOÀN TẤT!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()

