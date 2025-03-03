import os
import cv2
import csv
import re
import numpy as np  # Thêm import numpy để sử dụng np.median
from scipy.stats import mode

# Đường dẫn thư mục chứa ảnh
input_folder = "D:\\A_LastYear\\DOAN_TOTNGHIEP\\SORTWARE\\detect_color\\Detect_knn\\output_2"
output_csv = "D:\\A_LastYear\\DOAN_TOTNGHIEP\\SORTWARE\\detect_color\\Detect_knn\\data.csv"

# Tạo file CSV và ghi header nếu file chưa tồn tại
if not os.path.exists(output_csv):
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Hue", "Saturation", "Label"])

# Hàm trích xuất số frame từ tên file để sắp xếp file đúng thứ tự
def extract_frame_number(file_name):
    """Trích xuất số frame từ tên file (ví dụ: frame_1, frame_10)."""
    match = re.search(r'frame_(\d+)', file_name)
    return int(match.group(1)) if match else float('inf')  # Số lớn để đẩy file không hợp lệ xuống cuối

# Đọc danh sách ảnh trong thư mục và sắp xếp
image_list = sorted(os.listdir(input_folder), key=lambda x: (extract_frame_number(x), x))


def get_hsv_mode(image_path):
    """Lọc các pixel có Saturation thấp và tính giá trị Hue và Saturation trung vị."""
    image = cv2.imread(image_path)
    image = cv2.resize(image, (100, 100))
    image = cv2.GaussianBlur(image, (5, 5), 0)
    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Lọc những pixel có Value > 50 và Saturation > 50
    mask = (hsv_image[..., 1] > 50) & (hsv_image[..., 2] > 50)
    filtered_hue = hsv_image[..., 0][mask]
    filtered_saturation = hsv_image[..., 1][mask]
    
    if len(filtered_hue) == 0:  # Nếu không có pixel hợp lệ, trả về giá trị mặc định
        return 0, 0
    
    h_median = int(np.median(filtered_hue))
    s_median = int(np.median(filtered_saturation))
    
    return h_median, s_median


while True:
    # Hiển thị trạng thái và yêu cầu nhập quy tắc gán nhãn
    print("\nDanh sách ảnh có thể gán nhãn (hiển thị tối đa 10 ảnh đầu):")
    for i, img_name in enumerate(image_list[:10]):
        print(f"{i + 1}. {img_name}")
    print("...")

    print("\nNhập quy tắc gán nhãn (theo dãy tên ảnh):")
    start_name = input("Nhập tên ảnh bắt đầu (ví dụ: square_1_blue_frame_1.jpg): ").strip()
    end_name = input("Nhập tên ảnh kết thúc (ví dụ: square_1_blue_frame_100.jpg): ").strip()
    label = input("Nhập nhãn màu (r, g, b, y, o, w): ").strip()

    # Kiểm tra đầu vào hợp lệ
    if label not in ['r', 'g', 'b', 'y', 'o', 'w']:
        print("Nhãn màu không hợp lệ! Vui lòng nhập lại.")
        continue

    if start_name not in image_list or end_name not in image_list:
        print("Tên ảnh không tồn tại trong thư mục! Vui lòng nhập lại.")
        continue

    start_idx = image_list.index(start_name)
    end_idx = image_list.index(end_name) + 1
    selected_images = image_list[start_idx:end_idx]
    
    base_name = "_".join(start_name.split("_")[:3])
    filtered_images = [img for img in selected_images if img.startswith(base_name)]

    with open(output_csv, mode='a', newline='') as file:
        writer = csv.writer(file)
        for img_name in filtered_images:
            img_path = os.path.join(input_folder, img_name)
            h_mode, s_mode = get_hsv_mode(img_path)
            writer.writerow([h_mode, s_mode, label])
            print(f"Đã gán nhãn: {img_name} -> H={h_mode}, S={s_mode}, Label={label}")

    cont = input("Bạn có muốn tiếp tục? (y/n): ").lower()
    if cont != 'y':
        break

print("Hoàn tất quá trình gán nhãn!")
