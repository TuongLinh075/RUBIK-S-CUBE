import cv2

# Đọc ảnh Rubik
image = cv2.imread('rubik.jpg')

# Danh sách tọa độ các ô
square_coords = [
    (410, 200, 500, 280),  # Ô 1
    (560, 200, 650, 280),  # Ô 2
    (700, 200, 790, 280),  # Ô 3
    (410, 340, 500, 420),  # Ô 4
    (560, 340, 650, 420),  # Ô 5
    (700, 340, 790, 420),  # Ô 6
    (410, 480, 500, 560),  # Ô 7
    (560, 480, 650, 560),  # Ô 8
    (700, 480, 790, 560),  # Ô 9
]

# Hiển thị từng ô vuông
for i, (x1, y1, x2, y2) in enumerate(square_coords):
    # Trích xuất ô vuông từ tọa độ
    square = image[y1:y2, x1:x2]
    
    # Hiển thị ô vuông
    cv2.imshow(f'Square {i+1}', square)
    cv2.waitKey(500)  # Hiển thị mỗi ô trong 500ms

# Đóng tất cả cửa sổ sau khi hiển thị
cv2.destroyAllWindows()
