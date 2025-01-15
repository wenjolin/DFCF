# 引入必要模組
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from transformers import TFViTModel, ViTImageProcessor

tf.config.run_functions_eagerly(True)
# 設定與資料路徑
train_dir = ""
test_dir = ""
categories = ["real", "fake"]

# 使用 Hugging Face 提供的 ViT 圖像處理器進行輸入預處理
image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

def preprocess_images(image_batch):
    """將圖像批次預處理為 ViT 的輸入格式"""
    # 將圖像轉換為 TensorFlow 可接受的格式
    processed_images = image_processor(images=image_batch, return_tensors="tf")['pixel_values']
    return processed_images

# ========= 資料增強與預處理 =========
# 定義訓練集的圖像增強策略
train_datagen = ImageDataGenerator(
    rotation_range=30,  # 隨機旋轉角度範圍
    width_shift_range=0.2,  # 隨機水平平移
    height_shift_range=0.2,  # 隨機垂直平移
    brightness_range=[0.8, 1.2],  # 隨機調整亮度
    zoom_range=0.2,  # 隨機縮放
    horizontal_flip=True,  # 隨機水平翻轉
    rescale=1.0 / 255  # 將像素值歸一化到 [0, 1]
)

# 定義測試集的圖像處理策略（僅做標準化）
validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

# 建立訓練資料生成器
train_generator = train_datagen.flow_from_directory(
    train_dir,  # 訓練集資料夾路徑
    target_size=(224, 224),  # 將圖像調整為 224x224
    batch_size=32,  # 每批次處理 32 張圖片
    class_mode='binary'  # 二分類標籤模式
)

# 建立測試資料生成器
validation_generator = validation_datagen.flow_from_directory(
    test_dir,  # 測試集資料夾路徑
    target_size=(224, 224),  # 將圖像調整為 224x224
    batch_size=32,  # 每批次處理 32 張圖片
    class_mode='binary'  # 二分類標籤模式
)

# 加載 ViT 模型
vit_model = TFViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
vit_model.trainable = False

# 建立自定義分類器
class CustomModel(tf.keras.Model):
    def __init__(self, vit_model):
        super(CustomModel, self).__init__()
        self.vit_model = vit_model
        self.global_pooling = GlobalAveragePooling1D()
        self.dense1 = Dense(128, activation="relu")
        self.output_layer = Dense(1, activation="sigmoid")

    def call(self, inputs):
        # 進行預處理
        preprocessed_input = preprocess_images(inputs)
        # 提取特徵
        vit_output = self.vit_model(preprocessed_input).last_hidden_state
        # 全局平均池化
        pooled_features = self.global_pooling(vit_output)
        # 全連接層
        dense_output = self.dense1(pooled_features)
        return self.output_layer(dense_output)

# 定義完整模型
model = CustomModel(vit_model)

# 編譯模型，僅訓練分類層（凍結 ViT）
model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

# 訓練分類層
model.fit(
    train_generator,  # 使用訓練集生成器
    validation_data=validation_generator,  # 使用測試集生成器
    epochs=10,  # 訓練 10 個回合
    steps_per_epoch=len(train_generator),  # 每個回合的步數
    validation_steps=len(validation_generator)  # 驗證的步數
)

# ========= 全模型微調 =========
# 解凍 ViT 模型參數，進行全模型微調
vit_model.trainable = True

# 使用更低的學習率進行微調
model.compile(optimizer=Adam(learning_rate=0.0001), loss="binary_crossentropy", metrics=["accuracy"])

# 微調全模型
model.fit(
    train_generator,  # 使用訓練集生成器
    validation_data=validation_generator,  # 使用測試集生成器
    epochs=10,  # 訓練 10 個回合
    steps_per_epoch=len(train_generator),  # 每個回合的步數
    validation_steps=len(validation_generator)  # 驗證的步數
)

# ========= 評估模型性能 =========
# 在測試集上評估模型
test_loss, test_acc = model.evaluate(validation_generator)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# 預測測試集的標籤
y_true = validation_generator.classes  # 真實標籤
y_pred = (model.predict(validation_generator) > 0.5).astype(int)  # 將預測值轉換為二分類標籤

# 計算並打印混淆矩陣
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# 計算並打印各種評估指標
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")

# ========= 儲存模型 =========
# 將訓練好的模型保存為 H5 格式
model.save("vit_deepfake_model.h5")
print("Model has been saved.")
