import cv2
import numpy as np
import joblib
from time import sleep
import Adafruit_PCA9685
# import torch
import copy
import numpy as np
# import torch.nn as nn
# import torch.nn.functional as F
import time
import socket
import json
from scipy.stats import mode
# Khởi tạo PCA9685
pwm = Adafruit_PCA9685.PCA9685(busnum=1)
pwm.set_pwm_freq(50)  # Tần số 50 Hz

# load mo hinh knn
model_file = "/home/linh/Desktop/DoAn/Final/knn20_color_model_new.pkl"
knn = joblib.load(model_file)

current_angles = [0] * 16  
backlash = 4            
backlash_threshold = 1    
delay_step = 0.15
delay_big_step = 0.5
apetor = 1    
start = 45
end = 113

# Thêm vào đầu file, sau các import
u_bandau_moves = ["xoay_U", "xoay_U_p", "xoay_U_2", "xoay_U_p_2"]
d_bandau_moves = ["xoay_D", "xoay_D_p", "xoay_D_2", "xoay_D_p_2"]

'''
--------------------------------------------------------------------------s
DETECT PROCESS
--------------------------------------------------------------------------
'''
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

# Hàm lấy mode của H và S trong một vùng
def get_mode_hs(hsv_frame, x, y, size=30):
    region = hsv_frame[y:y+size, x:x+size]  # Lấy vùng ảnh
    h_values = region[:, :, 0].flatten()  # Lấy kênh H
    s_values = region[:, :, 1].flatten()  # Lấy kênh S

    h_mode = mode(h_values, axis=None, keepdims=True).mode[0]
    s_mode = mode(s_values, axis=None, keepdims=True).mode[0]
    
    return int(h_mode), int(s_mode)  # Trả về giá trị H và S

# print trang thai rubik
def print_rubiks_cube(state):
    # faces_order = ['up', 'left', 'front', 'right', 'back', 'down']
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
    for row in grid:
        print(' '.join(row))
# chuyen doi trang thai ve vector dau vao
def convert_to_vector(state):
    color_to_number = {
        'y': 0,  # U
        'w': 1,   # D
        'b': 2,    # L
        'g': 3,   # R
        'o': 4,  # B
        'r': 5      # F
    }
    faces_order = ['up', 'down', 'left', 'right', 'back', 'front']
    vector_input = []
    for face in faces_order:
        vector_input.append([color_to_number[color] for color in state[face]])
    print("th1",vector_input)
    print('--------------------------------------------')
    new_index_1 = [8,7,6,5,4,3,2,1,0]
    new_index_2 = [2,5,8,1,4,7,0,3,6]
    new_index_3 = [6,3,0,7,4,1,8,5,2]
    new_index_4 = [8,7,6,5,4,3,2,1,0]
    for index, face in enumerate(vector_input):
        if index == 2: 
            new_face = [face[i] for i in new_index_2]
            vector_input[index] = new_face
        if index == 3:
            new_face = [face[i] for i in new_index_3]
            vector_input[index] = new_face
        if index == 4:
            new_face = [face[i] for i in new_index_4]
            vector_input[index] = new_face
        if index == 1: 
            new_face = [face[i] for i in new_index_1]
            vector_input[index] = new_face
    return vector_input

'''
------------------------------------------------------------------------
SERVO PROCESS
------------------------------------------------------------------------
'''
def angle_to_ticks(angle):
    pulse = 0.5 + (angle / 180.0) * 2.0  # Tính pulse width (ms)
    ticks = int((pulse / 20.0) * 4096)   # Chuyển đổi sang ticks
    return ticks

# Hàm điều khiển servo với bù rơ được tối ưu hóa
def set_servo_angle(channel, target_angle,flag=True):
    global current_angles
    current_angle = current_angles[channel]  # Lấy góc hiện tại của servo

    angle_difference = abs(target_angle - current_angle)
    #print(f"Kênh: {channel}, Current: {current_angle}, Target: {target_angle}, Backlash: {backlash},angle_difference:{angle_difference}")
    if flag and angle_difference > backlash_threshold:
        if target_angle == 0:
            target_angle -= backlash if current_angle > target_angle else 0
        elif target_angle == 180:
            if channel == 4:
                target_angle += 9 
            elif channel != 4:
                target_angle += backlash 
        elif target_angle == 90:
            if current_angle > target_angle:
                target_angle -= backlash
            elif current_angle < target_angle:
                target_angle += backlash

    target_angle = max(0, min(190, target_angle))
    ticks = angle_to_ticks(target_angle)
    #print(f"Kênh: {channel}, Góc: {target_angle}, Ticks: {ticks}")
    #print("------------------------------------------------------")
    pwm.set_pwm(channel, 0, ticks)
    current_angles[channel] = target_angle

# đặt góc ban đầu
def goc_bandau():
    for channel in [4, 5, 6, 7]:
        set_servo_angle(channel, 90, flag=False)
    for channel in [0, 1, 2, 9]:
        set_servo_angle(channel, start)
    sleep(1)
    set_servo_angle(0, end + apetor)
    set_servo_angle(9, end )
    sleep(1)
    set_servo_angle(1, end )
    set_servo_angle(2, end + apetor)
    sleep(1)
# ham dieu khien
def move_servos(commands, delay):
    for number_index in commands:
        if len(number_index) == 3:
            channel,angle,flag = number_index
            set_servo_angle(channel, angle,flag)
        else:
            channel,angle = number_index
            set_servo_angle(channel,angle)
    sleep(delay)
    print(f"delay {delay}")

def Move_X(flag = False): # back
    move_servos([(1, start), (9, start)], delay=0.15)
    move_servos([(5,180,flag),(6,0,flag)],delay= 0.5)

def Move_x(flag = False): # front
    move_servos([(5,90,flag),(6,90,flag)],delay= 0.5)
    move_servos([(5,0,flag),(6,180,flag)],delay= 0.5)

def Move_Y(flag = False): #left
    move_servos([(5,90,flag),(6,90,flag)],delay= 0.5)
    move_servos([(1,end + apetor),(9,end + apetor)],delay= 0.15)
    move_servos([(0, start), (2, start)], delay=0.15)
    move_servos([(4,180),(7,0,flag)],delay= 0.5)

def Move_y(flag = False): #right
    move_servos([(4,90),(7,90,flag)],delay= 0.5)
    move_servos([(4,0),(7,180,flag)],delay= 0.5)

def Move_Z(flag = False): # up
    move_servos([(4,90),(7,90,flag)],delay= 0.2)
    move_servos([(5,0,flag),(6,180,flag)],delay= 0.3)
    move_servos([(0,end ),(2,end )],delay= 0.2)
    move_servos([(1, start), (9, start)], delay=0.15) 

def Move_z(flag= False): #d
    move_servos([(5,90,flag),(6,90,flag)],delay= 0.3)
    move_servos([(5,180,flag),(6,0,flag)],delay= 0.3)

def bandau(flag = False):
    move_servos([(5,90,flag),(6,90,flag)],delay= 0.3)
    move_servos([(5,0,flag),(6,180,flag)],delay= 0.3)
    move_servos([(1,end),(9,end)],delay= 0.15)
    move_servos([(0, start), (2, start)], delay=0.2) 
    move_servos([(5,90,flag),(6,90,flag)],delay= 0.3)
    move_servos([(0,end ),(2,end )],delay= 0.2)

def F_Face():
    move_servos([(6,180)],delay=0.5)
    move_servos([(0,start)],delay=0.15)
    move_servos([(6,90)],delay=0.15)
    move_servos([(0,end)],delay=0.2)

def f_Face():
    move_servos([(6,0)],delay=0.5)
    move_servos([(0,start)],delay=0.15)
    move_servos([(6,90)],delay=0.15)
    move_servos([(0,end)],delay=0.2)

def L_Face():
    move_servos([(4,180)],delay=0.5)
    move_servos([(1,start)],delay=0.15)
    move_servos([(4,90)],delay=0.15)
    move_servos([(1,end)],delay=0.2)

def l_Face():
    move_servos([(4,0)],delay=0.5)
    move_servos([(1,start)],delay=0.15)
    move_servos([(4,90)],delay=0.15)
    move_servos([(1,end)],delay=0.18)

def B_Face():
    move_servos([(5,180)],delay=0.5)
    move_servos([(2,start)],delay=0.15)
    move_servos([(5,90)],delay=0.15)
    move_servos([(2,end)],delay=0.2)

def b_Face():
    move_servos([(5,0)],delay=0.5)
    move_servos([(2,start)],delay=0.15)
    move_servos([(5,90)],delay=0.15)
    move_servos([(2,end)],delay=0.2)

def R_Face():
    move_servos([(7,180)],delay=0.5)
    move_servos([(9,start)],delay=0.15)
    move_servos([(7,90)],delay=0.15)
    move_servos([(9,end)],delay=0.2)

def r_Face():
    move_servos([(7,0)],delay=0.5)
    move_servos([(9,start)],delay=0.15)
    move_servos([(7,90)],delay=0.15)
    move_servos([(9,end)],delay=0.2)


def U_to_F(flag = False):
    move_servos([(1, end + apetor ), (9, end + apetor)], delay=0.15)
    move_servos([(0, start), (2, start)], delay=0.18)
    move_servos([(4, 180), (7, 0,flag)], delay=0.5)
    move_servos([(0, end), (2, end)], delay=0.18)
    move_servos([(1, start), (9, start)], delay=0.18)
    move_servos([(4, 90), (7, 90,flag)], delay=0.5)
    move_servos([(1, end + apetor), (9, end+apetor)], delay=0.2)

def D_to_F(flag = False):
    move_servos([(1, end + apetor ), (9, end + apetor)], delay=0.15)
    move_servos([(0, start), (2, start)], delay=0.18)
    move_servos([(4, 0), (7, 180,flag)], delay=0.5)
    move_servos([(0, end), (2, end)], delay=0.18) 
    move_servos([(1, start), (9, start)], delay=0.18)
    move_servos([(4, 90), (7, 90,flag)], delay=0.5)
    move_servos([(1, end + apetor), (9, end+apetor)], delay=0.2)
'''
==============================
========HÀM XOAY RUBIK========
==============================
'''
'''
---------- MAIN------------
'''

def execute_moves(moves, face_functions):
    # Đổi chỉ số: U -> F, D -> B, F -> D, B -> U
    face_indices_U = {'xoay_U': 'xoay_F', 'xoay_D': 'xoay_B', 
                      'xoay_F': 'xoay_D', 'xoay_B': 'xoay_U', 
                      'xoay_L': 'xoay_L', 'xoay_R': 'xoay_R',
                      'xoay_U_p': 'xoay_F_p', 'xoay_D_p': 'xoay_B_p', 
                      'xoay_F_p': 'xoay_D_p', 'xoay_B_p': 'xoay_U_p', 
                      'xoay_L_p': 'xoay_L_p', 'xoay_R_p': 'xoay_R_p'
                      }
    # đổi chỉ số: D -> F, U -> B, F -> U, B -> D
    face_indices_D = {'xoay_D': 'xoay_F', 'xoay_U': 'xoay_B', 
                      'xoay_F': 'xoay_U', 'xoay_B': 'xoay_D', 
                      'xoay_L': 'xoay_L', 'xoay_R': 'xoay_R',
                      'xoay_D_p': 'xoay_F_p', 'xoay_U_p': 'xoay_B_p', 
                      'xoay_F_p': 'xoay_U_p', 'xoay_B_p': 'xoay_D_p', 
                      'xoay_L_p': 'xoay_L_p', 'xoay_R_p': 'xoay_R_p'
                      }
    i = 0
    while i < len(moves):
        print(i)
        move = moves[i]
        if move == 'xoay_U' or move == 'xoay_U_p':
            U_to_F()
            sleep(0.1)
            moves = [face_indices_U.get(m, m) for m in moves]
        elif move == 'xoay_D' or move == 'xoay_D_p':
            D_to_F()
            sleep(0.1)
            moves = [face_indices_D.get(m, m) for m in moves]
        move_new = moves[i]
        if move_new in face_functions:
            move_name = face_functions[move_new]
            move_name()
        i += 1

def main():
    ############################################################################
    HOST = '0.0.0.0'  # Lắng nghe tất cả các IP
    PORT = 64432

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)

    print(f"Server đang lắng nghe trên {HOST}:{PORT}")
    conn, addr = server_socket.accept()
    print(f"Kết nối từ {addr}")

    #########################################################################
    color = {
        'red': (0, 0, 204),
        'orange': (0, 165, 255),
        'yellow': (0, 255, 255),
        'green': (0, 255, 0),
        'blue': (255, 0, 0),
        'white': (255, 255, 255),
        'pink': (255,0,255)
    }
    
    cap = cv2.VideoCapture(0)  

    label_points = [
        (210, 155), (280, 155), (360, 155),
        (210, 245), (280, 245), (360, 245),
        (210, 330), (280, 330), (360, 330)
    ]
    
    state = {
        'up': ['white'] * 9,
        'right': ['white'] * 9,
        'front': ['white'] * 9,
        'down': ['white'] * 9,
        'left': ['white'] * 9,
        'back': ['white'] * 9
    }
    
    # đua servo ve vitri ban dau 
    goc_bandau()
    Move_X()  
    total_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break 
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        current_state = []

        for i, (x, y) in enumerate(label_points):
            h_mode, s_mode = get_mode_hs(hsv_frame, x, y)  # Lấy mode H và S
            color_name = color_detect(h_mode, s_mode)  # Dự đoán màu
            rect_color = (255, 255, 255) if color_name == 'white' else color[color_name]
            cv2.rectangle(frame, (x, y), (x + 35, y + 35), rect_color, -1)  # Vẽ ô màu
            current_state.append(color_name[0]) 

        cv2.imshow('Color Detection', frame)      
        k = cv2.waitKey(1) & 0xFF
        if k == ord('r'):
            state['right'] = current_state
            print("Lưu mặt R")
            Move_x()
        if k == ord('l'):
            state['left'] = current_state
            print("Luu mat L")
            Move_Y()
        if k == ord('b'):
            state['back'] = current_state
            print("Luu mat B")
            Move_y()
        if k == ord('f'):
            state['front'] = current_state
            print("Luu mat F")
            Move_Z()
        if k == ord('u'):
            state['up'] = current_state
            print("Luu mat U")
            Move_z()
        if k == ord('d'):
            state['down'] = current_state
            print("Luu mat D")
            bandau()
        if k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Final Rubik's Cube state:")
    print(f"state la gi {state}")
    print_rubiks_cube(state)
    vector = convert_to_vector(state)
    print(f" VECTO INPUT: {vector}")

    data = conn.recv(1024)  # Nhận tín hiệu ban đầu từ laptop
    if data:
        print("Tín hiệu nhận được từ laptop.")
        time.sleep(0.05)  # Chờ 5 giây trước khi gửi
        conn.sendall(json.dumps(vector).encode('utf-8'))

    data = conn.recv(1024)
    if data:
        solution = json.loads(data.decode('utf-8'))  # Giải mã chuỗi JSON
        print(f"Chuỗi A nhận được: {solution}")

        face_functions = {
            "xoay_F": F_Face,
            "xoay_F_p": f_Face,
            "xoay_R": R_Face,
            "xoay_R_p": r_Face,
            "xoay_L": L_Face,
            "xoay_L_p": l_Face,
            "xoay_B": B_Face,
            "xoay_B_p": b_Face,
        }
        if solution:
            time_xoay = time.time()
            solution_a = solution['solutions']
            print("so buoc",len(solution_a))
            print(f'solution_a: {solution_a}')
            print(f"newsolution {solution_a}")

            if solution_a:
                execute_moves(solution_a, face_functions)
                time_end_xoay = time.time()
                t1 = time_end_xoay - time_xoay
                print("thoi gian xoay",t1)

            conn.sendall(b"done")
            print("Đã gửi tín hiệu hoàn tất xoay")
        else:
            print("OH NO ! NO SOLUTION RUBIK")
    else:
        print("Không nhận được dữ liệu.")
    total_end_time = time.time()
    t = total_end_time - total_time
    print(f" total_time: {t}")
    conn.close()
    server_socket.close()
if __name__ == "__main__":
    main()