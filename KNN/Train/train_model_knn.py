import csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Đọc dữ liệu từ tệp CSV
csv_file = "D:\\A_LastYear\\DOAN_TOTNGHIEP\\SORTWARE\\detect_color\\Detect_knn\\data.csv"
data = []
labels = []

with open(csv_file, mode='r') as file:
    reader = csv.reader(file)
    for row in reader:
        h, s, label = int(row[0]), int(row[1]), row[2]
        data.append([h, s])
        labels.append(label)

# Chuyển dữ liệu thành numpy array
X = np.array(data)      # Features (h, s)
y = np.array(labels)    # Labels (màu)

# Huấn luyện mô hình KNN
knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X, y)

# Lưu mô hình vào file
model_file = "knn20_color_model_new.pkl"
joblib.dump(knn, model_file)
print(f"Model đã được lưu vào {model_file}")
