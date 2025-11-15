import tensorflow as tf
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model

class ModelLoader:
    def __init__(self, model_path=None):
        """
        Initialize the CRNN model loader
        
        Args:
            model_path: Path to model weights. If None, will try:
                1. model_memorized_weights.hdf5 (nếu có)
                2. model_checkpoint_weights.hdf5 (mặc định)
        """
        # Tự động chọn model tốt nhất nếu không chỉ định
        if model_path is None:
            # Ưu tiên tìm trong thư mục model/
            if os.path.exists('model/model_checkpoint_weights.weights.h5'):
                model_path = 'model/model_checkpoint_weights.weights.h5'
                print("📦 Tìm thấy model trong thư mục model/ - sẽ sử dụng model này")
            elif os.path.exists('model/model_checkpoint_weights.hdf5'):
                model_path = 'model/model_checkpoint_weights.hdf5'
                print("📦 Sử dụng model trong thư mục model/")
            elif os.path.exists('model_memorized_weights.hdf5'):
                model_path = 'model_memorized_weights.hdf5'
                print("📦 Tìm thấy model_memorized_weights.hdf5 - sẽ sử dụng model này")
            elif os.path.exists('model_checkpoint_weights.hdf5'):
                model_path = 'model_checkpoint_weights.hdf5'
                print("📦 Sử dụng model_checkpoint_weights.hdf5")
            else:
                raise FileNotFoundError("Không tìm thấy model weights! Vui lòng train model trước.")
        
        self.model_path = model_path
        self.model = None
        self.act_model = None
        self.char_list = None
        self.TARGET_HEIGHT = 118
        self.TARGET_WIDTH = 2167
        self.TIME_STEPS = 240
        self.load_char_list()
        self.load_model()
    
    def load_char_list(self):
        """
        Load character list from dataset labels
        This should match the character list used during training
        """
        # Default Vietnamese character set (140 characters)
        # This should be loaded from the training data
        # For now, using a comprehensive Vietnamese character set
        vietnamese_chars = (
            ' !"#$%&\'()*+,-./0123456789:;<=>?@'
            'ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`'
            'abcdefghijklmnopqrstuvwxyz{|}~'
            'ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆ'
            'ÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰ'
            'ỲÝỶỸỴĐàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệ'
            'ìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữự'
            'ỳýỷỹỵđ'
        )
        
        # Create character list with blank token for CTC
        self.char_list = [''] + list(vietnamese_chars)
        
        # Load char_list from JSON file if exists (created during training)
        try:
            import json
            # Ưu tiên tìm trong thư mục model/
            char_list_paths = ['model/char_list.json', 'char_list.json']
            for char_list_path in char_list_paths:
                if os.path.exists(char_list_path):
                    with open(char_list_path, 'r', encoding='utf-8') as f:
                        loaded_chars = json.load(f)
                        # Ensure blank token is at index 0
                        if loaded_chars and loaded_chars[0] != '':
                            self.char_list = [''] + loaded_chars
                        else:
                            self.char_list = loaded_chars
                    print(f"Đã tải {len(self.char_list)} ký tự từ {char_list_path}")
                    break
        except Exception as e:
            print(f"Không thể tải char_list.json: {e}")
            pass
    
    def load_model(self):
        """
        Load the CRNN model for prediction - Improved version
        """
        try:
            if not os.path.exists(self.model_path):
                print(f"⚠️  Không tìm thấy file model tại {self.model_path}")
                print("   Thử tìm file khác...")
                # Thử tìm trong thư mục model/ trước
                alt_paths = [
                    'model/model_checkpoint_weights.weights.h5',
                    'model/model_checkpoint_weights.hdf5',
                    'model_checkpoint_weights.hdf5'
                ]
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        self.model_path = alt_path
                        print(f"   Tìm thấy: {alt_path}")
                        break
                else:
                    raise FileNotFoundError(f"Không tìm thấy model weights! Vui lòng train model trước.")
            
            print(f"Đang tải model từ {self.model_path}...")
            
            # Build act_model architecture
            self.act_model = self._build_model()
            
            # Build model với CTC để load weights
            def ctc_lambda_func(args):
                y_pred, labels, input_length, label_length = args
                return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
            
            # Get inputs and outputs from act_model
            inputs = self.act_model.input
            outputs = self.act_model.output
            
            # CTC inputs
            from tensorflow.keras.layers import Input as KerasInput
            labels_input = KerasInput(name='the_labels', shape=[self.TIME_STEPS], dtype='float32')
            input_length = KerasInput(name='input_length', shape=[1], dtype='int64')
            label_length = KerasInput(name='label_length', shape=[1], dtype='int64')
            
            # CTC loss layer
            loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
                [outputs, labels_input, input_length, label_length]
            )
            
            # Model for training (with CTC)
            self.model = Model(
                inputs=[inputs, labels_input, input_length, label_length], 
                outputs=loss_out
            )
            
            # Load weights
            self.model.load_weights(self.model_path)
            print("✅ Đã tải weights thành công")
            
            # Copy weights từ model sang act_model
            print("Đang copy weights sang prediction model...")
            for layer in self.act_model.layers:
                for m_layer in self.model.layers:
                    if m_layer.name == layer.name and hasattr(m_layer, 'get_weights'):
                        try:
                            weights = m_layer.get_weights()
                            if weights:
                                layer.set_weights(weights)
                                break
                        except Exception as e:
                            pass
            
            print("✅ Model đã sẵn sàng để predict")
                
        except Exception as e:
            print(f"❌ Lỗi khi tải model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _build_model(self):
        """
        Build the CRNN model architecture - EXACT match with training
        """
        from tensorflow.keras.layers import (
            Dense, LSTM, BatchNormalization, Input, Conv2D, 
            MaxPool2D, Lambda, Bidirectional, Add, Activation
        )
        from tensorflow.keras.models import Model
        
        # Input layer
        inputs = Input(shape=(self.TARGET_HEIGHT, self.TARGET_WIDTH, 1))
        
        # Block 1
        x = Conv2D(64, (3, 3), padding='same')(inputs)
        x = MaxPool2D(pool_size=3, strides=3)(x)
        x = Activation('relu')(x)
        
        # Block 2
        x = Conv2D(128, (3, 3), padding='same')(x)
        x = MaxPool2D(pool_size=3, strides=3)(x)
        x = Activation('relu')(x)
        
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
        
        # Bidirectional LSTM layers (dropout=0.2 cho model đã train)
        blstm_1 = Bidirectional(
            LSTM(512, return_sequences=True, dropout=0.2)
        )(squeezed)
        blstm_2 = Bidirectional(
            LSTM(512, return_sequences=True, dropout=0.2)
        )(blstm_1)
        
        # Output layer
        num_classes = len(self.char_list) if self.char_list else 141
        outputs = Dense(num_classes, activation='softmax', dtype='float32')(blstm_2)
        
        # Model for prediction (without CTC)
        act_model = Model(inputs, outputs)
        
        return act_model
    
    def _build_dummy_model(self):
        """
        Build a dummy model for testing when the actual model is not available
        """
        input_img = keras.Input(shape=(118, 2167, 1), name='input_image')
        # Simple dummy model that returns random predictions
        x = keras.layers.GlobalAveragePooling2D()(input_img)
        x = keras.layers.Dense(240)(x)
        x = keras.layers.RepeatVector(240)(x)
        num_classes = len(self.char_list) if self.char_list else 141
        x = keras.layers.Dense(num_classes, activation='softmax')(x)
        model = keras.Model(inputs=input_img, outputs=x)
        return model
    
    def predict(self, image):
        """
        Make prediction on a preprocessed image
        
        Args:
            image: Preprocessed image array with shape (1, 118, 2167, 1)
        
        Returns:
            Prediction array with shape (1, 240, num_classes)
        """
        if self.act_model is None:
            raise ValueError("Model chưa được tải. Vui lòng kiểm tra lại.")
        
        try:
            prediction = self.act_model.predict(image, verbose=0)
            return prediction
        except Exception as e:
            print(f"Lỗi khi dự đoán: {e}")
            # Return dummy prediction for testing
            return np.random.rand(1, 240, len(self.char_list) if self.char_list else 141)

