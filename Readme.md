# 🌊 Underwater Wireless Sensor Network (UWSN) Simulator

Dự án mô phỏng và tối ưu hóa tuổi thọ mạng cảm biến dưới nước

## 📋 Mục lục

- [Giới thiệu](#giới-thiệu)
- [Tính năng](#tính-năng)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt](#cài-đặt)
- [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
- [Tham số cấu hình](#tham-số-cấu-hình)
- [Kết quả](#kết-quả)


## 🎯 Giới thiệu

Dự án này mô phỏng một mạng cảm biến dưới nước (UWSN) trong không gian 3D, sử dụng:

- **Clustering và chọn cluster head (CH)**: Phân cụm các sensor nodes dựa trên năng lượng và vị trí
- **Các thuật toán tìm đường: GA, PSO, Greedy**
- **3D Visualization**: Hiển thị trực quan mạng với tương tác động


## ✨ Tính năng

### 1. Tạo dữ liệu đầu vào
- Sinh nodes phân bố đều trong không gian 3D (400×400×400m)
- Thử nghiệm nhiều kích thước mạng: 150-550 nodes
- Mỗi kích thước mạng có 10 bộ dữ liệu khác nhau

### 2. Phân cụm và visualization
- Phân cụm tự động với K-means
- Chọn Cluster Head theo năng lượng
- Visualization 3D mô hình phân cụm:
  - Xoay 360° (kéo chuột trái)
  - Zoom in/out (cuộn chuột hoặc phím +/-)
  - Pan (kéo chuột phải)
  - Hover hiển thị thông tin node

### 3. Phân tích kết quả
- So sánh tuổi thọ mạng (số chu kỳ hoàn thành)
- Tỷ lệ nodes còn sống
- Hiệu suất theo kích thước mạng

## 📁 Cấu trúc dự án

```
project/
│
├── data/
│   ├── input_data_evenly_distributed/    # Dữ liệu đầu vào
│   │   ├── nodes_150/
│   │   │   ├── nodes_150_1.json
│   │   │   ├── nodes_150_2.json
│   │   │   └── ...
│   │   ├── nodes_200/
│   │   ├── nodes_250/
│   │   └── ...
│   │
│   ├── output_data_kmeans/               # Kết quả phân cụm minh họa
│   │   ├── nodes_150_1.json
│   │   └── ...
│   │
│   └── draw_output_kmeans/               # Hình ảnh visualization minh họa phân cụm 
│       ├── nodes_150_1.png
│       └── ...
│
├── kaggle/
│   └── clustering.py                     # Module phân cụm
|   |__ algorthms                         # Module thuật toán định tuyến
|   |     |__ga.py
│   |     |__greedy.py
|   |     |__pso.py
│   |___compare_routing.py                # Script so sánh kết quả hội tụ giữa các thuật toán
│   |___compute.py                        # Module tính toán thời gian, năng lượng
│   |___simulate_routing.py               # Script chạy mô phỏng các bước phân cụm, định      |   |                                     # tuyến cho AUV di chuyển thu thập dữ liệu trong    |   |                                     # mạng
|   |___output
|       |___draw_hoitu
|       |    |____draw_output              # lưu ảnh biểu đồ so sánh độ hội tụ
|       |    |____nodes_150                # Chứa file json ghi lại kết quả định tuyến các lần
|       |     ...                          # lặp của từng thuật toán trên 10 bộ dữ liệu
|       |    |____draw_chart_hoitu.ipynb   # Vẽ biểu đồ và lưu vào draw_hoitu
|       |___results_routing                # Kết quả mô phỏng AUV thu thập trong mạng tới khi
|                                          # mạng sập
|            
├── create_input.ipynb                    # Notebook tạo dữ liệu
├── visualize_clustering.py               # Script phân cụm & vẽ
├── visualize_results.py                  # Script phân tích kết quả
└── README.md                             # File này
```

## 💻 Yêu cầu hệ thống

### Phần mềm
- Python 3.8+
- Jupyter Notebook (tùy chọn)

### Thư viện Python
```
numpy >= 1.20.0
matplotlib >= 3.3.0
scikit-learn >= 0.24.0
scipy >= 1.6.0
mplcursors >= 0.5.0
Pillow >= 8.0.0
```

## 🚀 Cài đặt

### Bước 1: Clone repository
```bash
git clone https://github.com/ttmanh03/IT4906.git
cd IT4906
```

### Bước 2: Tạo môi trường ảo (khuyến nghị)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Bước 3: Cài đặt dependencies
```bash
pip install numpy matplotlib scikit-learn scipy mplcursors Pillow
```

### Bước 4: Tạo thư mục dữ liệu
```bash
mkdir -p data/input_data_evenly_distributed
mkdir -p data/output_data_kmeans
mkdir -p data/draw_output_kmeans
```

## 📖 Hướng dẫn sử dụng

### 1️⃣ Tạo dữ liệu đầu vào

Mở và chạy notebook `create_input.ipynb` 


**Output**: 9 thư mục × 10 files = 90 bộ dữ liệu

### 2️⃣ Phân cụm và visualization

Chỉnh sửa đường dẫn trong `visualize_clustering.py`:

```python
# Dòng 16-18
input_folder = "your/path/to/input_data_evenly_distributed/nodes_150"
output_folder = "your/path/to/output_data_kmeans"
draw_folder = "your/path/to/draw_output_kmeans"
```

Chạy script:
```bash
python visualize_clustering.py
```

**Chức năng**:
- Đọc tất cả file JSON trong thư mục input
- Phân cụm với K-means 
- Chọn Cluster Head dựa trên năng lượng
- Xuất kết quả JSON với thông tin cụm
- Vẽ biểu đồ 3D interactive
- Lưu ảnh PNG 

**Điều khiển visualization**:
- **Chuột trái + kéo**: Xoay biểu đồ 360°
- **Cuộn chuột / +/-**: Zoom in/out
- **Chuột phải + kéo**: Di chuyển (pan)
- **Phím R**: Reset về góc nhìn mặc định
- **Hover**: Hiển thị thông tin node



## ⚙️ Tham số cấu hình

### Trong `visualize_clustering.py`

```python
# Không gian mạng
space_size = 400        # Kích thước không gian (m)
base_station = (200, 200, 400)  # Vị trí Base Station

# Cảm biến
r_sen = 60              # Bán kính cảm biến (m)
energy_node = 100       # Năng lượng ban đầu (J)

# Phân cụm
max_cluster_size = 20   # Số nodes tối đa trong 1 cụm
min_cluster_size = 5    # Số nodes tối thiểu trong 1 cụm
```

### Format dữ liệu đầu vào

**nodes_X_Y.json**:
```json
[
    {
        "id": 0,
        "x": 15.234,
        "y": 22.456,
        "z": 178.901,
        "energy_residual": 100.0,
        "energy_node": 100.0
    },
    ...
]
```

### Format dữ liệu đầu ra phân cụm

**nodes_X_Y.json** (trong output_data_kmeans):
```json
{
    "0": {
        "nodes": [5, 12, 23, 45, ...],
        "center": [120.5, 180.3, 200.7],
        "cluster_head": 12
    },
    "1": {
        "nodes": [1, 8, 15, 29, ...],
        "center": [250.1, 150.8, 180.2],
        "cluster_head": 8
    },
    ...
}
```

### Format kết quả mô phỏng

**result_nodes_X_Y.json**:
```json
{
    "input_file": "nodes_150_1.json",
    "initial_total_nodes": 150,
    "cycles_completed": 1239,
    "final_alive_nodes": 27,
    "final_alive_ratio": 0.1,
    "timestamp": "2025-01-13 14:30:25"
}
```

## 📊 Kết quả
### So sánh mức độ hiệu quả giữa các thuật toán
  - Tuổi thọ mạng là tiêu chí quan trọng nhất để đánh giá và qua đó, ta thấy: GA > PSO > Greedy

### Ví dụ kết quả visualization

**Biểu đồ 3D Interactive**:
- Nodes được tô màu theo cluster
- Cluster Head là hình vuông đen
- Base Station là tam giác xanh lá
- Đường nối từ member đến CH

**Phân tích hiệu suất**:
- Mạng 150 nodes: ~1200-1300 cycles
- Mạng 200 nodes: ~1100-1250 cycles
- Mạng 300+ nodes: ~900-1100 cycles
- Tỷ lệ sống cuối: < 10% (ngưỡng mạng sập)

### Insights

1. **Số nodes ↑ → Tuổi thọ mạng ↓**: 
   - Nhiều nodes = nhiều truyền thông = tiêu hao năng lượng nhanh hơn

2. **Hiệu suất (cycles/node)**:
   - Mạng nhỏ (~150 nodes): 8-9 cycles/node
   - Mạng lớn (~500 nodes): 2-3 cycles/node
   - Trade-off giữa coverage và tuổi thọ

3. **Khả năng mở rộng**:
   - Thuật toán hoạt động tốt với mạng 150-550 nodes
   - Scalability tốt với việc tăng nodes

