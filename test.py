from os import listdir
import os
import re
root_dir = os.path.dirname(os.path.abspath(__file__))
checkpoint_dir = os.path.join(root_dir, "checkpoint")
raw_folder = os.path.join(root_dir, "data")

os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(raw_folder, exist_ok=True)

import cv2
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import random
from keras.models import  load_model
import sys
import time

cap = cv2.VideoCapture(0)

# Dinh nghia class
# class_name = ['00000','10000','20000','50000','100000']

def get_num_classes(data_path):
    classes = [folder for folder in os.listdir(data_path) 
               if os.path.isdir(os.path.join(data_path, folder)) and not folder.startswith('.')]
    print(f"Số lượng lớp (class): {len(classes)}")
    print(f"Tên các lớp: {classes}")
    return len(classes), classes

num_classes, class_name = get_num_classes(raw_folder)

def get_model():
    model_vgg16_conv = VGG16(weights='imagenet', include_top=False)

    # Dong bang cac layer
    for layer in model_vgg16_conv.layers:
        layer.trainable = False

    # Tao model
    input = Input(shape=(128, 128, 3), name='image_input')
    output_vgg16_conv = model_vgg16_conv(input)

    # Them cac layer FC va Dropout
    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax', name='predictions')(x)

    # Compile
    my_model = Model(inputs=input, outputs=x)
    my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return my_model

# Load weights model da train
def get_best_weight(weights_folder=checkpoint_dir):
    # Lấy tất cả file có định dạng .keras trong thư mục checkpoint
    weight_files = [f for f in os.listdir(weights_folder) if f.endswith('.keras')]

    best_acc = -1
    best_file = None

    for file in weight_files:
        # Trích xuất val_accuracy từ tên file: weights-10-0.95.keras
        match = re.match(r"weights-\d{2}-(\d+\.\d+)\.keras", file)
        if match:
            acc = float(match.group(1))
            if acc > best_acc:
                best_acc = acc
                best_file = file

    if best_file:
        return os.path.join(weights_folder, best_file)
    else:
        return None


my_model = get_model()
best_weight = get_best_weight(checkpoint_dir)
if best_weight:
    print("Đang load model tốt nhất:", best_weight)
    my_model.load_weights(best_weight)
else:
    print("Không tìm thấy file weight tốt nhất!")
    exit()


while(True):
    # Capture frame-by-frame
    #

    ret, image_org = cap.read()
    if not ret:
        continue
    image_org = cv2.resize(image_org, dsize=None,fx=0.5,fy=0.5)
    
    # Resize
    image = image_org.copy()
    image = cv2.resize(image, dsize=(128, 128))
    image = image.astype('float')*1./255
    # Convert to tensor
    image = np.expand_dims(image, axis=0)

    # Predict
    predict = my_model.predict(image)
    print("This picture is: ", class_name[np.argmax(predict[0])], (predict[0]))
    print(np.max(predict[0],axis=0))
    if (np.max(predict)>=0.8) and (np.argmax(predict[0])!=0):


        # Show image
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        fontScale = 1.5
        color = (0, 255, 0)
        thickness = 2

        cv2.putText(image_org, class_name[np.argmax(predict)], org, font,
                    fontScale, color, thickness, cv2.LINE_AA)

    cv2.imshow("Picture", image_org)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

