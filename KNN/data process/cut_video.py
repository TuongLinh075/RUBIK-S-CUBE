import cv2
import os

# T1: CẮT VIDEO THÀNH CÁC ẢNH

input = "D:\\A_LastYear\\DOAN_TOTNGHIEP\\SORTWARE\\detect_color\\Detect_knn\\video_2"
output = "D:\\A_LastYear\\DOAN_TOTNGHIEP\\SORTWARE\\detect_color\\Detect_knn\\output"
num_frame = 60

# Tạo folder lưu ảnh nếu chưa có
if not os.path.exists(output):
    os.makedirs(output)
    
for i in os.listdir(input):
    video_path = os.path.join(input, i)
    video_name = os.path.splitext(i)[0]

    # Mở video
    cap = cv2.VideoCapture(video_path)
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frame // num_frame)
    
    frame_count = 0
    k = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % step == 0 and k < num_frame:
            output_path = os.path.join(output, f"{video_name}_frame_{k + 1}.jpg")
            cv2.imwrite(output_path, frame)
            k += 1
        frame_count += 1
    cap.release()

print("OK")


