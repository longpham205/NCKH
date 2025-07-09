from os import listdir
import os
import datetime
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
import random
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

root_dir = os.path.dirname(os.path.abspath(__file__))
checkpoint_dir = os.path.join(root_dir, "checkpoint")
raw_folder = os.path.join(root_dir, "data")

os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(raw_folder, exist_ok=True)

# ========================== Xử lý dữ liệu ==========================

def save_data(raw_folder=raw_folder):
    dest_size = (128, 128)
    print("Bắt đầu xử lý ảnh...")

    pixels = []
    labels = []

    for folder in listdir(raw_folder):
        if folder != '.DS_Store':
            print("Folder =", folder)
            folder_path = os.path.join(raw_folder, folder)
            for file in listdir(folder_path):
                if file != '.DS_Store':
                    print("File =", file)
                    img_path = os.path.join(folder_path, file)
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"⚠️ Không thể đọc ảnh: {img_path}")
                        continue
                    pixels.append(cv2.resize(img, dsize=(128, 128)))
                    labels.append(folder)

    pixels = np.array(pixels)
    labels = np.array(labels)

    encoder = LabelBinarizer()
    labels = encoder.fit_transform(labels)
    print(labels)

    with open(os.path.join(checkpoint_dir, 'pix.data'), 'wb') as file:
        pickle.dump((pixels, labels), file)

def load_data():
    with open(os.path.join(checkpoint_dir, 'pix.data'), 'rb') as file:
        pixels, labels = pickle.load(file)
    print(pixels.shape)
    print(labels.shape)
    return pixels, labels

# ========================== Load dữ liệu ==========================
save_data()
X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
print(X_train.shape)
print(y_train.shape)

# ========================== Tạo mô hình VGG ==========================

def get_num_classes(data_path):
    classes = [folder for folder in os.listdir(data_path) 
               if os.path.isdir(os.path.join(data_path, folder)) and not folder.startswith('.')]
    print(f"Số lượng lớp (class): {len(classes)}")
    print(f"Tên các lớp: {classes}")
    return len(classes), classes

num_classes, class_names = get_num_classes(raw_folder)

def get_model():
    model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
    for layer in model_vgg16_conv.layers:
        layer.trainable = False

    input = Input(shape=(128, 128, 3), name='image_input')
    output_vgg16_conv = model_vgg16_conv(input)

    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax', name='predictions')(x)

    my_model = Model(inputs=input, outputs=x)
    my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return my_model

vggmodel = get_model()

# === Lưu nhiều checkpoint, sau đó sẽ lọc giữ lại 3 cái tốt nhất ===
filepath = os.path.join(checkpoint_dir, "weights-{epoch:02d}-{val_accuracy:.2f}.keras")
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=False, mode='max')
callbacks_list = [checkpoint]

# ========================== Image Augmentation ==========================
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.1,
    rescale=1./255,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.2, 1.5],
    fill_mode="nearest"
)

aug_val = ImageDataGenerator(rescale=1./255)

# ========================== Huấn luyện ==========================
vgghist = vggmodel.fit(
    aug.flow(X_train, y_train, batch_size=64),
    epochs=20,
    validation_data=aug.flow(X_test, y_test, batch_size=64),
    callbacks=callbacks_list
)

vggmodel.save(os.path.join(checkpoint_dir, "vggmodel.h5"))

# ========================== Giữ lại 3 checkpoint tốt nhất ==========================
def cleanup_checkpoints(keep=3):
    files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.keras')]
    val_acc_pattern = re.compile(r"weights-\d{2}-(\d+\.\d+)\.keras")
    
    scored_files = []
    for file in files:
        match = val_acc_pattern.search(file)
        if match:
            val_acc = float(match.group(1))
            scored_files.append((file, val_acc))
    
    # Sắp xếp theo val_accuracy giảm dần
    scored_files.sort(key=lambda x: x[1], reverse=True)
    
    # Giữ lại 3 file đầu, xóa phần còn lại
    for file, _ in scored_files[keep:]:
        path_to_remove = os.path.join(checkpoint_dir, file)
        os.remove(path_to_remove)
        print(f" Đã xóa: {file}")

cleanup_checkpoints(keep=3)

# ========================== Vẽ biểu đồ ==========================
def plot_model_history(model_history, acc='accuracy', val_acc='val_accuracy'):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
    axs[0].plot(range(1, len(model_history.history[acc]) + 1), model_history.history[acc])
    axs[0].plot(range(1, len(model_history.history[val_acc]) + 1), model_history.history[val_acc])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(model_history.history[acc]) + 1))
    axs[0].legend(['train', 'val'], loc='best')

    axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
    axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(model_history.history['loss']) + 1))
    axs[1].legend(['train', 'val'], loc='best')

    plt.tight_layout()
    plt.show()

    # === Lưu ảnh vào chart/<ngày-tháng>/ ===
    today = datetime.datetime.now().strftime("%d-%m-%Y")
    chart_dir = os.path.join(root_dir, "chart", today)
    os.makedirs(chart_dir, exist_ok=True)
    count = len([f for f in os.listdir(chart_dir) if f.endswith('.png')])
    filename = f"{count + 1}.png"
    save_path = os.path.join(chart_dir, filename)
    fig.savefig(save_path)
    print(f" Đã lưu biểu đồ vào: {save_path}")

# Gọi hàm vẽ biểu đồ
plot_model_history(vgghist)
