import cv2
import os

# Đường dẫn đến folder chứa ảnh gốc và folder đích
input_folder = "D:\\A_LastYear\\DOAN_TOTNGHIEP\\SORTWARE\\detect_color\\Detect_knn\\output"  # Folder chứa ảnh gốc
output_folder = "D:\\A_LastYear\\DOAN_TOTNGHIEP\\SORTWARE\\detect_color\\Detect_knn\\resize_out"  # Folder lưu ảnh mới

# Tạo folder mới nếu chưa tồn tại
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Lấy danh sách tất cả file ảnh trong folder đầu vào
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Xử lý và lưu ảnh vào folder mới
for image_file in image_files:
    # Đường dẫn ảnh gốc
    input_path = os.path.join(input_folder, image_file)
    
    # Đọc ảnh
    image = cv2.imread(input_path)
    if image is None:
        print(f"Không thể mở ảnh: {image_file}")
        continue
    
    # (Tùy chọn) Thay đổi kích thước hoặc xử lý ảnh nếu cần
    resized_image = cv2.resize(image, (1280, 720))  # Resize ảnh về 640x360
    
    # Đường dẫn lưu ảnh mới
    output_path = os.path.join(output_folder, image_file)
    
    # Lưu ảnh đã xử lý
    cv2.imwrite(output_path, resized_image)
    print(f"Đã lưu ảnh: {output_path}")

print("Hoàn thành xử lý và lưu ảnh vào folder mới.")
