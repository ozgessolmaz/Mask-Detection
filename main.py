import os
import glob
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# === 1. VERİ HAZIRLIK ===

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

# Verileri oku
with_mask = glob.glob("/Users/ozgesolmaz/Desktop/gykodev/data/with_mask/*.jpg")
without_mask = glob.glob("/Users/ozgesolmaz/Desktop/gykodev/data/without_mask/*.jpg")

with_mask_labeled = [(img, 1) for img in with_mask]
without_mask_labeled = [(img, 0) for img in without_mask]
all_data = with_mask_labeled + without_mask_labeled
labels = [label for _, label in all_data]

train_data, val_data = train_test_split(
    all_data, test_size=0.2, random_state=42, stratify=labels
)

def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = image / 255.0
    return image, label

train_paths = [x[0] for x in train_data]
train_labels = [x[1] for x in train_data]
val_paths = [x[0] for x in val_data]
val_labels = [x[1] for x in val_data]

train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))

train_ds = train_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

# === 2. MODEL OLUŞTUR & EĞİT ===

base_model = MobileNetV2(input_shape=(224, 224, 3),
                         include_top=False,
                         weights='imagenet')
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[early_stop]
)

# === 3. EĞİTİM GRAFİĞİ ===

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Train Accuracy')
plt.plot(epochs_range, val_acc, label='Val Accuracy')
plt.legend(loc='lower right')
plt.title('Model Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Train Loss')
plt.plot(epochs_range, val_loss, label='Val Loss')
plt.legend(loc='upper right')
plt.title('Model Loss')

plt.tight_layout()
plt.show()

# === 4. CONFUSION MATRIX ve METRİKLER ===

val_images = []
val_true_labels = []

for image, label in val_ds.unbatch():
    val_images.append(image.numpy())
    val_true_labels.append(label.numpy())

val_preds = model.predict(np.array(val_images))
val_preds_binary = (val_preds > 0.5).astype("int32")

cm = confusion_matrix(val_true_labels, val_preds_binary)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Mask", "Mask"],
            yticklabels=["No Mask", "Mask"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

print("Classification Report:")
print(classification_report(val_true_labels, val_preds_binary, target_names=["No Mask", "Mask"]))

# === 5. 5 ADET GÖRSEL ÜZERİNDE TAHMİN VE GÖSTERİM ===

indices = random.sample(range(len(val_images)), 5)
plt.figure(figsize=(15, 5))

for i, idx in enumerate(indices):
    img = val_images[idx]
    true_label = val_true_labels[idx]
    pred_label = val_preds_binary[idx][0]

    plt.subplot(1, 5, i+1)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Gerçek: {'Mask' if true_label==1 else 'No Mask'}\nTahmin: {'Mask' if pred_label==1 else 'No Mask'}")

plt.tight_layout()
plt.show()
# denem
