import numpy as np
import copy, random
from collections import deque
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os

# Khai báo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cube = [[0,0,0,0,0,0,0,0,0],  [1,1,1,1,1,1,1,1,1], [2,2,2,2,2,2,2,2,2],
        [3,3,3,3,3,3,3,3,3],  [4,4,4,4,4,4,4,4,4], [5,5,5,5,5,5,5,5,5]]
reward_done = 1
state_done = copy.deepcopy(cube)
history = []

def is_valid_move(new_move):
    if not history:
        return True
    if new_move == history[-1] + "_p" or (new_move.endswith("_p") and new_move[:-2] == history[-1]):
        return False
    if history.count(new_move) >= 3:
        return False
    return True

def reset():
    global cube, history
    cube = copy.deepcopy(state_done)
    history = []

def render():
    color_codes = {
        0: "\033[48;5;226m",  # U (màu vàng)
        1: "\033[48;5;15m",   # D (màu trắng)
        2: "\033[48;5;21m",   # L (màu xanh dương)
        3: "\033[48;5;46m",   # R (màu xanh lá)
        4: "\033[48;5;208m",  # B (màu cam)
        5: "\033[48;5;196m",  # F (màu đỏ)
        'reset': "\033[0m"
    }
    print("       " + color_codes[cube[0][0]] + "  " + color_codes[cube[0][1]] + "  " + color_codes[cube[0][2]] + "  " + color_codes['reset'])
    print("       " + color_codes[cube[0][3]] + "  " + color_codes[cube[0][4]] + "  " + color_codes[cube[0][5]] + "  " + color_codes['reset'])
    print("       " + color_codes[cube[0][6]] + "  " + color_codes[cube[0][7]] + "  " + color_codes[cube[0][8]] + "  " + color_codes['reset'])
    print("")
    print(color_codes[cube[2][0]] + "  " + color_codes[cube[2][1]] + "  " + color_codes[cube[2][2]] + "  " + color_codes['reset'], end=" ")
    print(color_codes[cube[5][0]] + "  " + color_codes[cube[5][1]] + "  " + color_codes[cube[5][2]] + "  " + color_codes['reset'], end=" ")
    print(color_codes[cube[3][0]] + "  " + color_codes[cube[3][1]] + "  " + color_codes[cube[3][2]] + "  " + color_codes['reset'], end=" ")
    print(color_codes[cube[4][0]] + "  " + color_codes[cube[4][1]] + "  " + color_codes[cube[4][2]] + "  " + color_codes['reset'])

    print(color_codes[cube[2][3]] + "  " + color_codes[cube[2][4]] + "  " + color_codes[cube[2][5]] + "  " + color_codes['reset'], end=" ")
    print(color_codes[cube[5][3]] + "  " + color_codes[cube[5][4]] + "  " + color_codes[cube[5][5]] + "  " + color_codes['reset'], end=" ")
    print(color_codes[cube[3][3]] + "  " + color_codes[cube[3][4]] + "  " + color_codes[cube[3][5]] + "  " + color_codes['reset'], end=" ")
    print(color_codes[cube[4][3]] + "  " + color_codes[cube[4][4]] + "  " + color_codes[cube[4][5]] + "  " + color_codes['reset'])

    print(color_codes[cube[2][6]] + "  " + color_codes[cube[2][7]] + "  " + color_codes[cube[2][8]] + "  " + color_codes['reset'], end=" ")
    print(color_codes[cube[5][6]] + "  " + color_codes[cube[5][7]] + "  " + color_codes[cube[5][8]] + "  " + color_codes['reset'], end=" ")
    print(color_codes[cube[3][6]] + "  " + color_codes[cube[3][7]] + "  " + color_codes[cube[3][8]] + "  " + color_codes['reset'], end=" ")
    print(color_codes[cube[4][6]] + "  " + color_codes[cube[4][7]] + "  " + color_codes[cube[4][8]] + "  " + color_codes['reset'])
    print("")
    print("       " + color_codes[cube[1][0]] + "  " + color_codes[cube[1][1]] + "  " + color_codes[cube[1][2]] + "  " + color_codes['reset'])
    print("       " + color_codes[cube[1][3]] + "  " + color_codes[cube[1][4]] + "  " + color_codes[cube[1][5]] + "  " + color_codes['reset'])
    print("       " + color_codes[cube[1][6]] + "  " + color_codes[cube[1][7]] + "  " + color_codes[cube[1][8]] + "  " + color_codes['reset'])

def xoay_U():
    global cube
    cube[0] = [cube[0][6], cube[0][3], cube[0][0],
               cube[0][7], cube[0][4], cube[0][1],
               cube[0][8], cube[0][5], cube[0][2]]
    temp = [cube[2][0], cube[2][1], cube[2][2]]
    cube[2][0] = cube[5][0]
    cube[2][1] = cube[5][1]
    cube[2][2] = cube[5][2]
    cube[5][0] = cube[3][0]
    cube[5][1] = cube[3][1]
    cube[5][2] = cube[3][2]
    cube[3][0] = cube[4][0]
    cube[3][1] = cube[4][1]
    cube[3][2] = cube[4][2]
    cube[4][0] = temp[0]
    cube[4][1] = temp[1]
    cube[4][2] = temp[2]

def xoay_U_p():
    for _ in range(3):
        xoay_U()

def xoay_D():
    global cube
    cube[1] = [cube[1][6], cube[1][3], cube[1][0],
               cube[1][7], cube[1][4], cube[1][1],
               cube[1][8], cube[1][5], cube[1][2]]
    temp = [cube[2][6], cube[2][7], cube[2][8]]
    cube[2][6] = cube[4][6]
    cube[2][7] = cube[4][7]
    cube[2][8] = cube[4][8]
    cube[4][6] = cube[3][6]
    cube[4][7] = cube[3][7]
    cube[4][8] = cube[3][8]
    cube[3][6] = cube[5][6]
    cube[3][7] = cube[5][7]
    cube[3][8] = cube[5][8]
    cube[5][6] = temp[0]
    cube[5][7] = temp[1]
    cube[5][8] = temp[2]

def xoay_D_p():
    for _ in range(3):
        xoay_D()

def xoay_R():
    global cube
    cube[3] = [cube[3][6], cube[3][3], cube[3][0],
               cube[3][7], cube[3][4], cube[3][1],
               cube[3][8], cube[3][5], cube[3][2]]
    temp = [cube[0][2], cube[0][5], cube[0][8]]
    cube[0][2] = cube[5][2]
    cube[0][5] = cube[5][5]
    cube[0][8] = cube[5][8]
    cube[5][2] = cube[1][2]
    cube[5][5] = cube[1][5]
    cube[5][8] = cube[1][8]
    cube[1][2] = cube[4][6]
    cube[1][5] = cube[4][3]
    cube[1][8] = cube[4][0]
    cube[4][0] = temp[2]
    cube[4][3] = temp[1]
    cube[4][6] = temp[0]

def xoay_R_p():
    for _ in range(3):
        xoay_R()

def xoay_L():
    global cube
    cube[2] = [cube[2][6], cube[2][3], cube[2][0],
               cube[2][7], cube[2][4], cube[2][1],
               cube[2][8], cube[2][5], cube[2][2]]
    temp = [cube[0][0], cube[0][3], cube[0][6]]
    cube[0][0] = cube[4][2]
    cube[0][3] = cube[4][5]
    cube[0][6] = cube[4][8]
    cube[4][2] = cube[1][6]
    cube[4][5] = cube[1][3]
    cube[4][8] = cube[1][0]
    cube[1][0] = cube[5][0]
    cube[1][3] = cube[5][3]
    cube[1][6] = cube[5][6]
    cube[5][0] = temp[0]
    cube[5][3] = temp[1]
    cube[5][6] = temp[2]

def xoay_L_p():
    for _ in range(3):
        xoay_L()

def xoay_F():
    global cube
    cube[5] = [cube[5][6], cube[5][3], cube[5][0],
               cube[5][7], cube[5][4], cube[5][1],
               cube[5][8], cube[5][5], cube[5][2]]
    temp = [cube[0][6], cube[0][7], cube[0][8]]
    cube[0][6] = cube[2][8]
    cube[0][7] = cube[2][5]
    cube[0][8] = cube[2][2]
    cube[2][2] = cube[1][0]
    cube[2][5] = cube[1][1]
    cube[2][8] = cube[1][2]
    cube[1][0] = cube[3][6]
    cube[1][1] = cube[3][3]
    cube[1][2] = cube[3][0]
    cube[3][0] = temp[0]
    cube[3][3] = temp[1]
    cube[3][6] = temp[2]

def xoay_F_p():
    for _ in range(3):
        xoay_F()

def xoay_B():
    global cube
    cube[4] = [cube[4][6], cube[4][3], cube[4][0],
               cube[4][7], cube[4][4], cube[4][1],
               cube[4][8], cube[4][5], cube[4][2]]
    temp = [cube[0][0], cube[0][1], cube[0][2]]
    cube[0][0] = cube[3][2]
    cube[0][1] = cube[3][5]
    cube[0][2] = cube[3][8]
    cube[3][2] = cube[1][8]
    cube[3][5] = cube[1][7]
    cube[3][8] = cube[1][6]
    cube[1][6] = cube[2][0]
    cube[1][7] = cube[2][3]
    cube[1][8] = cube[2][6]
    cube[2][0] = temp[0]
    cube[2][3] = temp[1]
    cube[2][6] = temp[2]

def xoay_B_p():
    for _ in range(3):
        xoay_B()

def render_number():
    print("         ", cube[0][0], cube[0][1], cube[0][2])
    print("         ", cube[0][3], cube[0][4], cube[0][5])
    print("         ", cube[0][6], cube[0][7], cube[0][8])
    print("")
    print(cube[2][0], cube[2][1], cube[2][2], "   ",
          cube[5][0], cube[5][1], cube[5][2], "   ",
          cube[3][0], cube[3][1], cube[3][2], "   ",
          cube[4][0], cube[4][1], cube[4][2])
    print(cube[2][3], cube[2][4], cube[2][5], "   ",
          cube[5][3], cube[5][4], cube[5][5], "   ",
          cube[3][3], cube[3][4], cube[3][5], "   ",
          cube[4][3], cube[4][4], cube[4][5])
    print(cube[2][6], cube[2][7], cube[2][8], "   ",
          cube[5][6], cube[5][7], cube[5][8], "   ",
          cube[3][6], cube[3][7], cube[3][8], "   ",
          cube[4][6], cube[4][7], cube[4][8])
    print("")
    print("         ", cube[1][0], cube[1][1], cube[1][2])
    print("         ", cube[1][3], cube[1][4], cube[1][5])
    print("         ", cube[1][6], cube[1][7], cube[1][8])

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


def xao_tron_rubik(k):
    state_taken = []
    move_taken = []
    for i in range(1):
        reset()
        for _ in range(k):
            while True:
                m = random.choice(moves)
                i = moves.index(m)
                move_name = move_abc[i]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
                if is_valid_move(move_name):
                    break
            m()
            history.append(move_name)
            if len(history) > 3:
                history.pop(0)
            state = copy.deepcopy(cube)
            state_taken.append(state)
            move_taken.append(i)
    return state_taken, move_taken


class ResidualBlock(nn.Module):
    def __init__(self, input_dim):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(input_dim, input_dim)
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.linear2 = nn.Linear(input_dim, input_dim)
        self.bn2 = nn.BatchNorm1d(input_dim)

    def forward(self, x):
        residual = x
        out = self.linear1(x)
        out = self.linear2(out)
        out += residual
        return out

class RubikModel(nn.Module):
    def __init__(self):
        super(RubikModel, self).__init__()
        self.linear1 = nn.Linear(324, 5000)
        self.bn1 = nn.BatchNorm1d(5000)
        self.linear2 = nn.Linear(5000, 1000)
        self.bn2 = nn.BatchNorm1d(1000)

        self.residual_blocks = nn.Sequential(
            ResidualBlock(1000),
            ResidualBlock(1000),
            ResidualBlock(1000),
            ResidualBlock(1000)
        )
        self.linear3 = nn.Linear(1000, 12)

    def forward(self, x):
        x = nn.functional.one_hot(x, num_classes=6).to(torch.float)
        x = x.reshape(-1, 324)
        x = F.relu(self.linear1(x))
        x = self.bn1(x)
        x = F.relu(self.linear2(x))
        x = self.bn2(x)
        x = self.residual_blocks(x)
        x = self.linear3(x)
        return x

class RubikDataset(Dataset):
    def __init__(self, step):
        self.state_taken, self.move_taken = xao_tron_rubik(step)

    def __len__(self):
        return len(self.state_taken)

    def __getitem__(self, idx):
        state = np.array(self.state_taken[idx]).reshape(-1, 54)
        move = np.array(self.move_taken[idx])

        # Chuyển đổi sang tensor
        state_tensor = torch.tensor(state, dtype=torch.long)
        move_tensor = torch.tensor(move, dtype=torch.long)

        return state_tensor.to(device), move_tensor.to(device)

model = RubikModel()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()
step = 26
epochs = 5200000
loss_history = []
state_taken, move_taken = xao_tron_rubik(1)
print(f"state_taken : {state_taken}")
print(f"move_taken : {move_taken}")
state_tensor = torch.tensor(state_taken, dtype=torch.long).to(device)
print(f"gia trị:  {state_tensor}")
move_tensor = torch.tensor(move_taken, dtype=torch.long).to(device)
print(f"move_tensor: {move_tensor}")
# Load checkpoint nếu có
# checkpoint_path = "/content/rubik_model_checkpoint.pth"
# start_epoch = 0

# if os.path.exists(checkpoint_path):
#     checkpoint = torch.load(checkpoint_path)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     start_epoch = checkpoint['epoch'] + 1  # Tiếp tục từ epoch tiếp theo
#     loss_history = checkpoint.get('loss_history', [])  # Load loss history nếu có
#     print(f"Loaded checkpoint from epoch {start_epoch}")
    
# for epoch in range(start_epoch, epochs):
#     state_taken, move_taken = xao_tron_rubik(step)
    
#     # Chuyển đổi toàn bộ dữ liệu thành tensor
#     state_tensor = torch.tensor(state_taken, dtype=torch.long).to(device)
#     print(f"gia trị {len(state_tensor)}")
#     move_tensor = torch.tensor(move_taken, dtype=torch.long).to(device)

#     # Huấn luyện mô hình với toàn bộ dữ liệu
#     output = model(state_tensor)
#     loss = loss_func(output, move_tensor)

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     # Lưu lại loss
#     total_loss = loss.item()
#     loss_history.append(total_loss)
#     print(f'Epoch [{epoch+1}], Loss: {total_loss:.4f}')
#     if (epoch + 1) % 100 == 0:
#         # Vẽ toàn bộ quá trình huấn luyện
#         plt.figure(figsize=(10, 5))
#         plt.plot(loss_history, label='Overall Loss')
#         plt.title('Overall Loss')
#         plt.xlabel('Epoch')
#         plt.ylabel('Loss')
#         plt.grid(True)
#         plt.legend()
#         plt.savefig("overall_loss.png")

#         # Vẽ chỉ 100 epochs gần nhất
#         plt.figure(figsize=(10, 5))
#         plt.plot(loss_history[-200:], label='Last 100 Epochs Loss')  # Vẽ 100 giá trị gần nhất
#         plt.title('Loss of Last 100 Epochs')
#         plt.xlabel('Epoch')
#         plt.ylabel('Loss')
#         plt.grid(True)
#         plt.legend()
#         plt.savefig("last_200_epochs_loss.png")

#         plt.show(block=False)
#         plt.pause(3)
#         plt.close()
#         # Lưu mô hình
#         torch.save({
#             'epoch': epochs,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss_history': loss_history,
#         }, 'rubik_model_checkpoint.pth')
# print("da hoan thanh train")
# plt.figure(figsize=(10, 5))
# plt.plot(loss_history)
# plt.title('Loss value')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.grid(True)
# plt.savefig("rubik.png")
# plt.close
