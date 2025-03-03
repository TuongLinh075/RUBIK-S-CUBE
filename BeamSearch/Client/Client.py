import torch
import random
import copy
import numpy as np
import copy, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
import time
import cv2
import socket
import json
import threading
import tkinter as tk
from tkinter import ttk
import  queue
"""
TCP
"""
model2 = torch.jit.load("D:\\A_LastYear\\DOAN_TOTNGHIEP\\SORTWARE\\detect_color\\final\\cube3.pth", map_location=torch.device('cpu'))
def xoay_U(a):
    a[0] =    [a[0][6], a[0][3], a[0][0],
               a[0][7], a[0][4], a[0][1],
               a[0][8], a[0][5], a[0][2]]
    # Lưu lại hàng đầu của mặt L
    temp = [a[2][0], a[2][1], a[2][2]]
    # L -> F
    a[2][0] = a[5][0]
    a[2][1] = a[5][1]
    a[2][2] = a[5][2]
    # F -> R
    a[5][0] = a[3][0]
    a[5][1] = a[3][1]
    a[5][2] = a[3][2]
    # R -> B
    a[3][0] = a[4][0]
    a[3][1] = a[4][1]
    a[3][2] = a[4][2]
    # B -> L
    a[4][0] = temp[0]
    a[4][1] = temp[1]
    a[4][2] = temp[2]

def xoay_U_p(a):
    for _ in range(3):
        xoay_U(a)
def xoay_D(a):
    a[1] =    [a[1][6], a[1][3], a[1][0],
               a[1][7], a[1][4], a[1][1],
               a[1][8], a[1][5], a[1][2]]

    # Lưu lại hàng cuối của mặt L
    temp = [a[2][6], a[2][7], a[2][8]]
    # L -> B
    a[2][6] = a[4][6]
    a[2][7] = a[4][7]
    a[2][8] = a[4][8]
    # B -> R
    a[4][6] = a[3][6]
    a[4][7] = a[3][7]
    a[4][8] = a[3][8]
    # R -> F
    a[3][6] = a[5][6]
    a[3][7] = a[5][7]
    a[3][8] = a[5][8]
    # F -> L
    a[5][6] = temp[0]
    a[5][7] = temp[1]
    a[5][8] = temp[2]

def xoay_D_p(a):
    for _ in range(3):
        xoay_D(a)
        
def xoay_R(a):
    a[3] =    [a[3][6], a[3][3], a[3][0],
               a[3][7], a[3][4], a[3][1],
               a[3][8], a[3][5], a[3][2]]
    # Lưu lại hàng cuối của mặt U
    temp = [a[0][2], a[0][5], a[0][8]]
    # U -> F
    a[0][2] = a[5][2]
    a[0][5] = a[5][5]
    a[0][8] = a[5][8]
    # F -> D
    a[5][2] = a[1][2]
    a[5][5] = a[1][5]
    a[5][8] = a[1][8]
    # D -> B
    a[1][2] = a[4][6]
    a[1][5] = a[4][3]
    a[1][8] = a[4][0]
    # B -> U
    a[4][0] = temp[2]
    a[4][3] = temp[1]
    a[4][6] = temp[0]

def xoay_R_p(a):
    for _ in range(3):
        xoay_R(a)

def xoay_L(a):
    a[2] =  [a[2][6], a[2][3], a[2][0],
             a[2][7], a[2][4], a[2][1],
             a[2][8], a[2][5], a[2][2]]
    # Lưu lại hàng đầu của mặt U
    temp = [a[0][0], a[0][3], a[0][6]]
    # U -> B
    a[0][0] = a[4][8]
    a[0][3] = a[4][5]
    a[0][6] = a[4][2]
    # B -> D
    a[4][2] = a[1][6]
    a[4][5] = a[1][3]
    a[4][8] = a[1][0]
    # D -> F
    a[1][0] = a[5][0]
    a[1][3] = a[5][3]
    a[1][6] = a[5][6]
    # F -> U
    a[5][0] = temp[0]
    a[5][3] = temp[1]
    a[5][6] = temp[2]

def xoay_L_p(a):
    for _ in range(3):
        xoay_L(a)

def xoay_F(a):
    a[5] = [a[5][6], a[5][3], a[5][0],
               a[5][7], a[5][4], a[5][1],
               a[5][8], a[5][5], a[5][2]]
    # Lưu lại hàng cuối của mặt U
    temp = [a[0][6], a[0][7], a[0][8]]
    # U -> L
    a[0][6] = a[2][8]
    a[0][7] = a[2][5]
    a[0][8] = a[2][2]
    # L -> D
    a[2][2] = a[1][0]
    a[2][5] = a[1][1]
    a[2][8] = a[1][2]
    # D -> R
    a[1][0] = a[3][6]
    a[1][1] = a[3][3]
    a[1][2] = a[3][0]
    # R -> U
    a[3][0] = temp[0]
    a[3][3] = temp[1]
    a[3][6] = temp[2]

def xoay_F_p(a):
    for _ in range(3):
        xoay_F(a)

def xoay_B(a):
    a[4] =  [a[4][6], a[4][3], a[4][0],
             a[4][7], a[4][4], a[4][1],
             a[4][8], a[4][5], a[4][2]]
    # Lưu lại hàng đầu của mặt U
    temp = [a[0][0], a[0][1], a[0][2]]
    # U -> R
    a[0][0] = a[3][2]
    a[0][1] = a[3][5]
    a[0][2] = a[3][8]
    # R -> D
    a[3][2] = a[1][8]
    a[3][5] = a[1][7]
    a[3][8] = a[1][6]
    # D -> L
    a[1][6] = a[2][0]
    a[1][7] = a[2][3]
    a[1][8] = a[2][6]
    # L -> U
    a[2][0] = temp[2]
    a[2][3] = temp[1]
    a[2][6] = temp[0]

def xoay_B_p(a):
    for _ in range(3):
        xoay_B(a)

def reset():
    global cube, history
    cube = copy.deepcopy(state_done)
    history = []
    return cube, history

state_done = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [2, 2, 2, 2, 2, 2, 2, 2, 2],
    [3, 3, 3, 3, 3, 3, 3, 3, 3],
    [4, 4, 4, 4, 4, 4, 4, 4, 4],
    [5, 5, 5, 5, 5, 5, 5, 5, 5]
]
moves = [
    xoay_R, xoay_R_p,
    xoay_L, xoay_L_p,
    xoay_F, xoay_F_p,
    xoay_B, xoay_B_p,
    xoay_U, xoay_U_p,
    xoay_D, xoay_D_p
]
move_abc = [
    'xoay_R', 'xoay_R_p',
    'xoay_L', 'xoay_L_p',
    'xoay_F', 'xoay_F_p',
    'xoay_B', 'xoay_B_p',
    'xoay_U', 'xoay_U_p',
    'xoay_D', 'xoay_D_p'
]
move_abc_nguoc = [
    'xoay_R_p', 'xoay_R',
    'xoay_L_p', 'xoay_L',
    'xoay_F_p', 'xoay_F',
    'xoay_B_p', 'xoay_B',
    'xoay_U_p', 'xoay_U',
    'xoay_D_p', 'xoay_D'
]
def is_solved(state):
    return all(len(set(face)) == 1 for face in state)

nguoc_face = {
    'R': 'L', 'L': 'R',
    'F': 'B', 'B': 'F',
    'U': 'D', 'D': 'U'
}
nguoc_xoay = {
    'xoay_R': 'xoay_R_p', 'xoay_R_p': 'xoay_R',
    'xoay_L': 'xoay_L_p', 'xoay_L_p': 'xoay_L',
    'xoay_F': 'xoay_F_p', 'xoay_F_p': 'xoay_F',
    'xoay_B': 'xoay_B_p', 'xoay_B_p': 'xoay_B',
    'xoay_U': 'xoay_U_p', 'xoay_U_p': 'xoay_U',
    'xoay_D': 'xoay_D_p', 'xoay_D_p': 'xoay_D'
}
def check_move(history, new_move):
    if not history:
        return True
    last_move = history[-1]
    if new_move == last_move + "_p" or (new_move.endswith("_p") and new_move[:-2] == last_move):
        return False
    face = new_move[5]
    count = 0
    for past_move in reversed(history):
        if past_move[5] == face:
            count += 1
        elif past_move[5] == nguoc_face[face]:
            break
    if count >= 2:
        return False
    return True

def xao_tron_rubik(k):
    state, history = reset()
    moves_performed = []
    for _ in range(k):
        while True:
            m = random.choice(moves)
            i = moves.index(m)
            move_name = move_abc[i]
            if check_move(history, move_name):
                break
        moves_performed.append(move_name)
        m(state)
        history.append(move_name)
        if len(history) > 3:
            history.pop(0)
        cube = copy.deepcopy(state)
    print("Chuỗi xoay đã thực hiện:", ", ".join(moves_performed))
    return cube
def xao_tron_rubik_theo_chuoi(commands):
    """
    Xáo trộn rubik theo chuỗi lệnh xoay người dùng cung cấp, ánh xạ tới các hàm trong moves.
    
    :param commands: Chuỗi các lệnh xoay (vd: "R L L' U' F").
    :return: Trạng thái cuối cùng của rubik sau các bước xoay.
    """
    state, history = reset()  # Khởi tạo trạng thái rubik
    moves_performed = []  # Danh sách các bước xoay đã thực hiện

    # Từ điển ánh xạ giữa lệnh và hàm xoay
    mapping = {
        "R": xoay_R, "R'": xoay_R_p,
        "L": xoay_L, "L'": xoay_L_p,
        "F": xoay_F, "F'": xoay_F_p,
        "B": xoay_B, "B'": xoay_B_p,
        "U": xoay_U, "U'": xoay_U_p,
        "D": xoay_D, "D'": xoay_D_p,
    }

    command_list = commands.split()  # Chuyển chuỗi lệnh thành danh sách

    for command in command_list:
        if command in mapping:  # Kiểm tra xem lệnh có trong mapping không
            move_func = mapping[command]  # Lấy hàm tương ứng
            move_func(state)  # Thực hiện xoay
            moves_performed.append(command)  # Lưu lệnh đã thực hiện
            history.append(command)  # Cập nhật lịch sử
            if len(history) > 3:
                history.pop(0)
        else:
            print(f"Lệnh không hợp lệ: {command}")
            continue

    print("Chuỗi xoay đã thực hiện:", ", ".join(moves_performed))
    return state

def quay_nguoc(move_name):
    if "_p" in move_name:
        return move_name.replace("_p", "")
    else:
        return move_name + "_p"
    
def check_history(history):
    aaa = []
    for move in history:
        if aaa:
            if move == quay_nguoc(aaa[-1]):
                aaa.pop()
                continue
            if len(aaa) >= 2 and move == aaa[-1] == aaa[-2]:
                aaa = aaa[:-2]
                aaa.append(quay_nguoc(move))
                continue
            if len(aaa) >= 3 and move == aaa[-1] == aaa[-2] == aaa[-3]:
                aaa = aaa[:-3]
                continue
            move1 = aaa[-1]
            move2 = move
            a = nguoc_face.get(move1[0]) == move2[0]
            if len(aaa) >= 2 and move == quay_nguoc(aaa[-2]) and a:
                aaa.pop()
                aaa.pop()
                continue
        aaa.append(move)
    return aaa


def tao_chuoi_xao_tron(scramble_length):
    moves = ['U', "U'", 'D', "D'", 'L', "L'", 'R', "R'", 'B', "B'", 'F', "F'"]
    nguoc_face = {
        'R': 'L', 'L': 'R',
        'F': 'B', 'B': 'F',
        'U': 'D', 'D': 'U'
    }
    
    def is_valid_move(scramble, new_move):
        if not scramble:
            return True
        last_move = scramble[-1]
        if new_move == last_move + "'" or (new_move.endswith("'") and new_move[:-1] == last_move):
            return False
        if len(scramble) >= 2 and new_move == scramble[-1] == scramble[-2]:
            return False
        if len(scramble) >= 2 and new_move[0] == nguoc_face.get(scramble[-2][0]) and new_move[0] == scramble[-1][0]:
            return False
        return True

    scramble_sequence = []
    while len(scramble_sequence) < scramble_length:
        move = random.choice(moves)
        if is_valid_move(scramble_sequence, move):
            scramble_sequence.append(move)
    
    return " ".join(scramble_sequence)

@torch.no_grad()
def beam_search_final(state, model, k, max_depth, device, skip_redundant_moves):
    model.to(device)
    model.eval()
    node, time_0 = 0, time.time()
    state = np.array(state).reshape(-1, 54)
    beam = [{"state": copy.deepcopy(state), "path": [], "value": 1.0}]

    for depth in range(max_depth + 1):
        print(f"Độ sâu hiện tại: {depth}")
        batch_x = np.zeros((len(beam), 54), dtype=np.int64)
        for i, c in enumerate(beam):
            c_path, state = c["path"], c["state"]
            if c_path:
                state_new = copy.deepcopy(state)
                state_2d = state_new.reshape((6, 9)).tolist()
                move_name = c_path[-1]
                moves[move_abc.index(move_name)](state_2d)
                node += 1
                if is_solved(state_2d):
                    return {'solutions': c_path, "num_nodes": node, "times": time.time() - time_0}
                state_2d = np.array(state_2d).flatten()
                state = state_2d
                c["state"] = state
            batch_x[i, :] = state

        if depth == max_depth:
            print("Không giải được")
            return None
        beam_next_depth = []
        if len(beam) < 2**17:
            batch_x = torch.from_numpy(batch_x).to(device)
            batch_p = model(batch_x)
            batch_p = torch.nn.functional.softmax(batch_p, dim=-1)
            batch_p = batch_p.detach().cpu().numpy()
        else:
            batch_p = [
                torch.nn.functional.softmax(model(torch.from_numpy(batch_x_mini).to(device))).to('cpu').detach().numpy()
                for batch_x_mini in np.split(batch_x, len(beam) // (2**16))
            ]
            batch_p = np.concatenate(batch_p)

        for i, c in enumerate(beam):
            c_path = c["path"]
            xs = batch_p[i, :]
            xs *= c["value"]

            for m, value in zip(move_abc_nguoc, xs):
                if c_path and skip_redundant_moves:
                    if m == nguoc_xoay[c_path[-1]]:
                        continue
                    if len(c_path) > 1 and c_path[-2] == c_path[-1] == m:
                        continue
                    if len(c_path) > 2 and m == nguoc_xoay[c_path[-2]] and c_path[-1][0] == nguoc_face.get(c_path[-2][0]):
                        continue
                beam_next_depth.append({
                    'state': copy.deepcopy(c['state']),
                    "path": c_path + [m],
                    "value": value,
                })

        beam = sorted(beam_next_depth, key=lambda x: -x["value"])
        beam = beam[:k]


class TimerWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Timer")
        self.label = ttk.Label(self.root, text="0.00 giây", font=("Arial", 24))
        self.label.pack(padx=30, pady=30)
        self.start_time = None
        self.running = False

    def start(self):
        self.start_time = time.time()
        self.running = True
        self.update_timer()

    def stop(self):
        self.running = False
        return time.time() - self.start_time

    def update_timer(self):
        if self.running:
            elapsed = time.time() - self.start_time
            self.label.config(text=f"{elapsed:.2f} giây")
            self.root.after(100, self.update_timer)

class TCPHandler:
    def __init__(self, host, port, message_queue):
        self.host = host
        self.port = port
        self.message_queue = message_queue
        self.socket = None
        self.running = True

    def connect(self):
        """Kết nối tới server."""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))
        self.socket.sendall("Tín hiệu kết nối từ laptop".encode('utf-8'))

    def listen(self, device, timer_window):
        """Lắng nghe dữ liệu từ server trong luồng riêng."""
        try:
            while self.running:
                data = self.socket.recv(1024)
                if data:
                    message = data.decode('utf-8')
                    self.message_queue.put(message)  # Đưa dữ liệu vào hàng đợi để xử lý
                    if message == "done":
                        elapsed = timer_window.stop()
                        print("Giải thành công!")
                        print(f"Tổng thời gian: {elapsed:.2f} giây")
                        timer_window.root.after(3000, timer_window.root.quit)
                    else:
                        vecto_input = json.loads(message)
                        print(f"VECTO INPUT nhận được: {vecto_input}")

                        # Tính toán giải pháp
                        solution = beam_search_final(
                            vecto_input, model2, k=2**11, max_depth=40, 
                            device=device, skip_redundant_moves=True
                        )
                        if solution:
                            self.socket.sendall(json.dumps(solution).encode('utf-8'))
                            print("Kết quả ===>", solution)
        except Exception as e:
            print(f"Lỗi trong luồng TCP: {e}")
        finally:
            self.running = False
            self.socket.close()

    def stop(self):
        """Dừng giao tiếp TCP."""
        self.running = False

def main():
    HOST = '192.168.38.110'  # Địa chỉ IP của Raspberry Pi
    PORT = 64432
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Khởi tạo giao diện Timer
    timer_window = TimerWindow()

    # Khởi tạo hàng đợi thông điệp
    message_queue = queue.Queue()

    # Khởi tạo xử lý TCP
    tcp_handler = TCPHandler(HOST, PORT, message_queue)
    tcp_handler.connect()

    # Bắt đầu đếm thời gian
    # timer_window.start()
# Bắt đầu đếm thời gian sau 1 giây
    timer_window.root.after(2000, timer_window.start)

    # Tạo luồng riêng cho giao tiếp TCP
    tcp_thread = threading.Thread(target=tcp_handler.listen, args=(device, timer_window))
    tcp_thread.daemon = True
    tcp_thread.start()

    # Chạy mainloop Tkinter
    timer_window.root.mainloop()

    # Dừng TCP khi thoát
    tcp_handler.stop()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Lỗi: {e}")