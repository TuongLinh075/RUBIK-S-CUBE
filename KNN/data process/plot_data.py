import csv
import matplotlib.pyplot as plt
import numpy as np

# Bản đồ nhãn thành màu
label_to_color = {
    'o': 'orange',
    'w': 'black',
    'r': 'red',
    'b': 'blue',
    'g': 'green',
    'y': 'yellow'
}

# Đọc dữ liệu từ file CSV
csv_file = "D:\\A_LastYear\\DOAN_TOTNGHIEP\\SORTWARE\\detect_color\\Detect_knn\\colors_data_new1.csv"
h_values, s_values, labels = [], [], []

with open(csv_file, mode='r') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        h, s, label = int(row[0]), int(row[1]), row[2]
        h_values.append(h)
        s_values.append(s)
        labels.append(label)

# Vẽ các điểm với màu nhãn
for h, s, label in zip(h_values, s_values, labels):
    plt.scatter(h, s, color=label_to_color[label], label=label, alpha=0.1, edgecolor='k', linewidth=0)

# Chỉ hiển thị mỗi nhãn một lần trong chú thích
handles, unique_labels = [], []
for label in set(labels):
    handles.append(plt.Line2D([], [], marker='o', color=label_to_color[label], linestyle='None'))
    unique_labels.append(label)

plt.legend(handles, unique_labels, title="Labels", loc='upper right')
plt.title("Hue-Saturation Space with Density")
plt.xlabel("Hue (H)")
plt.ylabel("Saturation (S)")
plt.grid(True)
plt.show()
