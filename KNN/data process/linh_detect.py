import cv2
import numpy as np
import joblib
from scipy.stats import mode

# Nạp mô hình đã huấn luyện
model_file = "/home/linh/Desktop/DoAn/detect_color/linh.pkl"
knn = joblib.load(model_file)

# Hàm phát hiện màu sử dụng mô hình
def color_detect(h, s):
    label = knn.predict([[h, s]])[0]  # Dự đoán nhãn
    color_map = {
        'r': 'red',
        'g': 'green',
        'b': 'blue',
        'y': 'yellow',
        'o': 'orange',
        'w': 'white'
    }
    return color_map.get(label, 'pink')  # Trả về tên màu tương ứng với nhãn

# Hàm lấy mode của H và S tại vị trí (x, y) trên ảnh HSV
def get_mode_hs(hsv_frame, x, y):

    region = hsv_frame[y-2:y+3, x-2:x+3]
    
    # Lọc pixels có S và V đủ lớn
    mask = (region[..., 1] > MIN_SATURATION) & (region[..., 2] > MIN_VALUE)
    filtered_h = region[..., 0][mask]
    filtered_s = region[..., 1][mask]
    
    if len(filtered_h) == 0:
        return 0, 0
        
    h_mode = int(mode(filtered_h).mode[0])
    s_mode = int(mode(filtered_s).mode[0])
    
    return h_mode, s_mode

def print_rubiks_cube(state):

   face_positions = {
       'up': (0, 1),
       'left': (1, 0),
       'front': (1, 1),
       'right': (1, 2),
       'back': (1, 3),
       'down': (2, 1)
   }
   
   grid = [[' ']*12 for _ in range(9)]
   for face, pos in face_positions.items():
       r_offset = pos[0] * 3
       c_offset = pos[1] * 3
       for i in range(3):
           for j in range(3):
               grid[r_offset + i][c_offset + j] = state[face][i*3 + j]
   
   print("\nRubik's Cube State:")
   for row in grid:
       print(' '.join(row))
def main():
   try:
       # Load mô hình
       knn = load_model(MODEL_PATH)
       
       # Khởi tạo camera
       cap = cv2.VideoCapture(0)
       if not cap.isOpened():
           raise ValueError("Không thể mở camera")
        # Vị trí các ô trên mặt Rubik
       label_points = [
           (212, 155), (280, 155), (355, 155),
           (212, 245), (280, 245), (355, 245),
           (212, 330), (280, 330), (355, 330)
       ]
        # Khởi tạo trạng thái
       state = {face: ['white'] * 9 for face in ['up', 'right', 'front', 'down', 'left', 'back']}
       
       while True:
           ret, frame = cap.read()
           if not ret:
               logging.error("Không thể đọc frame từ camera")
               break
           
           hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
           current_state = []
           
           # Xử lý từng điểm
           for x, y in label_points:
               h_mode, s_mode = get_mode_hs(hsv_frame, x, y)
               color_name = color_detect(knn, h_mode, s_mode)
               rect_color = RGB_COLORS[color_name]
               cv2.rectangle(frame, (x, y), (x + RECT_SIZE, y + RECT_SIZE), rect_color, -1)
               current_state.append(color_name[0])
           
           # Hiển thị frame
           cv2.imshow(WINDOW_NAME, frame)
           
           # Xử lý phím bấm
           key = cv2.waitKey(1) & 0xFF
           if key == ord('q'):
               break
           elif key in [ord(f) for f in 'urldtb']:
               face = {'u': 'up', 'r': 'right', 'l': 'left', 
                      'd': 'down', 'f': 'front', 'b': 'back'}[chr(key)]
               state[face] = current_state.copy()
               logging.info(f"{face.capitalize()} face saved.")
           elif key == ord('p'):
               print_rubiks_cube(state)
               
   except Exception as e:
       logging.error(f"Lỗi: {e}")
   finally:
       cap.release()
       cv2.destroyAllWindows()
       print("\nFinal Rubik's Cube state:")
       print_rubiks_cube(state)
if __name__ == '__main__':
   main()