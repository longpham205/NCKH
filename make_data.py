import numpy as np
import cv2
import time
import os
import os
root_dir = os.path.dirname(os.path.abspath(__file__))
raw_folder = os.path.join(root_dir, "data")
os.makedirs(raw_folder, exist_ok=True)

# Label: 00000 là ko cầm tiền, còn lại là các mệnh giá
label = "10000"

cap = cv2.VideoCapture(0)

min_img = 60
max_img = 1000

# Biến đếm, để chỉ lưu dữ liệu sau khoảng 60 frame, tránh lúc đầu chưa kịp cầm tiền lên
i=0
while(True):
    # Capture frame-by-frame
    #
    i+=1
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.resize(frame, dsize=None,fx=0.3,fy=0.3)

    # Hiển thị
    cv2.imshow('frame',frame)

    # Lưu dữ liệu
    if i>= min_img and i < max_img:
        print("Số ảnh capture = ",i-60)
        # Tạo thư mục nếu chưa có
        label_path = os.path.join(raw_folder, str(label))
        if not os.path.exists(label_path):
            os.makedirs(label_path)

        cv2.imwrite(label_path + "/" + label + "_" + str(i) + ".png",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    if i >= max_img:
        print("Đã chụp đủ ảnh. Dừng chương trình.")
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()