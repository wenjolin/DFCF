import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling1D, LayerNormalization, BatchNormalization
from tensorflow.keras.models import Model
# 優先使用實驗性版本的 AdamW，若不可用則退回 legacy 版本
try:
    from tensorflow.keras.optimizers.experimental import AdamW
except ImportError:
    from tensorflow.keras.optimizers.legacy import AdamW
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import get_custom_objects
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from transformers import TFViTModel, ViTImageProcessor
import seaborn as sns
import matplotlib.pyplot as plt

# ※【不啟用混合精度】（移除 set_global_policy），以避免因型態轉換問題影響 ViT 模型運算
# tf.keras.mixed_precision.set_global_policy("mixed_float16")
# tf.config.run_functions_eagerly(True)  # 如需除錯可啟用 eager 模式，但平常可關閉以提升效能

###############################################
# 1. 資料前處理與臉部裁剪
###############################################

# 路徑設定
original_train_dir = "/Users/0yuan_0124/train"
original_test_dir  = "/Users/0yuan_0124/test_T"
processed_train_dir = "/Users/0yuan_0124/processed_train"
processed_test_dir  = "/Users/0yuan_0124/processed_test"

os.makedirs(processed_train_dir, exist_ok=True)
os.makedirs(processed_test_dir, exist_ok=True)

def crop_faces(input_dir, output_dir):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    for root, _, files in os.walk(input_dir):
        for filename in files:
            img_path = os.path.join(root, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(faces) == 0:
                continue
            # 選取面積最大的臉
            faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
            x, y, w, h = faces[0]
            face = img[y:y+h, x:x+w]
            face = cv2.resize(face, (224, 224))
            rel_dir = os.path.relpath(root, input_dir)
            save_dir = os.path.join(output_dir, rel_dir)
            os.makedirs(save_dir, exist_ok=True)
            output_path = os.path.join(save_dir, filename)
            cv2.imwrite(output_path, face)

print("Processing training data...")
crop_faces(original_train_dir, processed_train_dir)
print("Processing testing data...")
crop_faces(original_test_dir, processed_test_dir)

###############################################
# 2. 數據生成與預處理
###############################################

# 加載 ViT 預處理器，這裡設定 do_rescale=True（會根據模型要求將像素值正規化）
image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", do_rescale=True)

def preprocess_images(images):
    images = np.array(images, dtype=np.float32)
    images = np.clip(images, 0, 255)
    processed = image_processor(images=images, return_tensors="tf").pixel_values
    # 確保最後一個維度為3
    if processed.shape[-1] != 3:
        processed = tf.transpose(processed, perm=[0, 1, 3, 2])
    processed = tf.where(tf.math.is_nan(processed), tf.zeros_like(processed), processed)
    return tf.cast(processed, tf.float16)  # 確保數據類型為 float16

class PreprocessedGenerator(tf.keras.utils.Sequence):
    def __init__(self, generator):
        self.generator = generator
        self.classes = generator.classes

    def __len__(self):
        return len(self.generator)

    def __getitem__(self, index):
        images, labels = self.generator[index]
        processed_images = preprocess_images(images)
        return processed_images, labels

# 數據增強設定
train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3]
)
validation_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    processed_train_dir, target_size=(224, 224), batch_size=32, class_mode='binary', color_mode='rgb'
)
validation_generator = validation_datagen.flow_from_directory(
    processed_test_dir, target_size=(224, 224), batch_size=32, class_mode='binary', color_mode='rgb'
)

train_generator = PreprocessedGenerator(train_generator)
validation_generator = PreprocessedGenerator(validation_generator)

# 計算類別權重（若資料不均衡可幫助訓練）
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

###############################################
# 3. 模型定義與微調（使用 ViT）
###############################################

# 加載 ViT 模型
vit_model = TFViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
vit_model.trainable = False

class CustomModel(tf.keras.Model):
    def __init__(self, vit_model):
        super(CustomModel, self).__init__()
        self.vit_model = vit_model
        self.norm = LayerNormalization()
        self.global_pooling = GlobalAveragePooling1D()
        self.dense1 = Dense(1024, activation="relu", kernel_regularizer=l2(0.01))
        self.bn1 = BatchNormalization()
        self.dropout1 = Dropout(0.4)
        self.dense2 = Dense(512, activation="relu", kernel_regularizer=l2(0.01))
        self.bn2 = BatchNormalization()
        self.dropout2 = Dropout(0.4)
        self.output_layer = Dense(1, activation="sigmoid", dtype=tf.float32)

    def call(self, inputs):
        # 將輸入轉換為 float32，確保 ViT 模型內部運算使用正確的型態
        inputs_fp32 = tf.cast(inputs, tf.float32)
        vit_output = self.vit_model(inputs_fp32).last_hidden_state
        norm_output = self.norm(vit_output)
        pooled_output = self.global_pooling(norm_output)
        x = self.dense1(pooled_output)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        return self.output_layer(x)

model = CustomModel(vit_model)

# 解凍 ViT 模型最後 24 層以進行微調
for layer in vit_model.layers[-24:]:
    layer.trainable = True

###############################################
# 4. 編譯與訓練
###############################################

# 註冊自訂優化器（解決序列化時找不到 AdamW 的問題）
get_custom_objects()['AdamW'] = AdamW
get_custom_objects()['adamw'] = AdamW

model.compile(
    optimizer=AdamW(learning_rate=1e-4, weight_decay=1e-4),
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
    metrics=["accuracy"]
)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
checkpoint = ModelCheckpoint('best_model', monitor='val_loss', save_best_only=True, verbose=1)

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=15,
    callbacks=[early_stopping, lr_scheduler, checkpoint],
    class_weight=class_weights
)

###############################################
# 5. 模型評估與儲存
###############################################

# 先建立模型圖（對 subclassed 模型需先呼叫 build 才能正確儲存）
#model.build(input_shape=(None, 224, 224, 3))

test_loss, test_acc = model.evaluate(validation_generator)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

y_true = validation_generator.classes
y_pred = (model.predict(validation_generator) > 0.5).astype(int)

target_names = ["real", "fake"]
report = classification_report(y_true, y_pred, target_names=target_names)
print("\nClassification Report:")
print(report)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
print(f"Accuracy: {acc * 100:.2f}%")
print(f"Precision: {prec * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")

# 儲存模型（使用 TensorFlow SavedModel 格式）
model.save("final_best_model", save_format="tf")
print("Best model has been saved.")
