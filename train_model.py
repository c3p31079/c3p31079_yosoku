import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import os

# データパス
data_path = "dataset"

# 画像読み込み
datagen = ImageDataGenerator(rescale=1/255, validation_split=0.2)

train = datagen.flow_from_directory(
    data_path, target_size=(224,224),
    class_mode='categorical', subset='training'
)
val = datagen.flow_from_directory(
    data_path, target_size=(224,224),
    class_mode='categorical', subset='validation'
)

# 転移学習
base = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

x = layers.Dense(128, activation='relu')(base.output)
x = layers.Dropout(0.2)(x)
output = layers.Dense(6, activation='softmax')(x)  # 6クラス(2部位×3劣化)

model = models.Model(base.input, output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 学習
history = model.fit(train, validation_data=val, epochs=10)

# 保存
model.save("swing_degradation_model.h5")
print("✅ Model saved as swing_degradation_model.h5")
