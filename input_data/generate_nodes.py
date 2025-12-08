import numpy as np
import json
import os

def generate_nodes(N, space_size=400, filename="nodes.json", initial_energy=100.0):
    # Sinh N node cảm biến ngẫu nhiên trong không gian 3D với residual energy
    
    # N: số lượng node
    # space_size: kích thước không gian
    # filename: tên file lưu
    # initial_energy: năng lượng ban đầu (E0)

    
    np.random.seed(0)  # để kết quả lặp lại
    node_positions = np.random.rand(N, 3) * space_size
    
    # Sinh residual energy ngẫu nhiên (50-100% của initial energy)
    np.random.seed(42)  # Seed khác để energy không phụ thuộc vào vị trí
    residual_energies = np.random.uniform(0.5 * initial_energy, initial_energy, N)

    data = []
    for i in range(N):
        data.append({
            "id": i,
            "x": float(node_positions[i][0]),
            "y": float(node_positions[i][1]),
            "z": float(node_positions[i][2]),
            # "residual_energy": float(residual_energies[i]),
            "residual_energy": float(initial_energy),
            "initial_energy": float(initial_energy)
        })

    
    os.makedirs("IT4906/input_data", exist_ok=True)
    filepath = f"IT4906/input_data/{filename}"
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Đã tạo file {filepath} chứa {N} node với residual energy.")
    print(f"Energy range: {residual_energies.min():.1f} - {residual_energies.max():.1f}")

# Sinh dữ liệu với energy information
def main():
    for N in [150, 200, 250, 300, 350, 400, 450, 500, 550]:
        generate_nodes(N, space_size=400, filename=f"nodes_{N}.json", initial_energy=100.0)

if __name__ == "__main__":
    main()