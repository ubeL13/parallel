import os
import re
import matplotlib.pyplot as plt

lab1_dir = "lab1/results"
lab2_dir = "lab2/results"
output_dir = "lab2/results"
os.makedirs(output_dir, exist_ok=True)

def read_size_and_time(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    match_size = re.search(r"Matrix A: (\d+)x(\d+)", content)
    match_time = re.search(r"Execution Time \(s\): ([\d\.]+)", content)
    if match_size and match_time:
        size = int(match_size.group(1)) 
        time = float(match_time.group(1))
        return size, time
    return None, None

times_lab1 = []
times_lab2 = []
sizes = []

idx = 1
while True:
    file1 = os.path.join(lab1_dir, f"time_{idx}.txt")
    file2 = os.path.join(lab2_dir, f"time_{idx}.txt")

    if not os.path.exists(file1) or not os.path.exists(file2):
        break

    size1, t1 = read_size_and_time(file1)
    size2, t2 = read_size_and_time(file2)

    if None not in (size1, t1, size2, t2) and size1 == size2:
        sizes.append(size1)
        times_lab1.append(t1)
        times_lab2.append(t2)

    idx += 1

plt.figure(figsize=(10, 6))
plt.plot(sizes, times_lab1, marker='o', label='Lab 1 (no OpenMP)')
plt.plot(sizes, times_lab2, marker='s', label='Lab 2 (with OpenMP)')
plt.title('Execution Time vs Matrix Size')
plt.xlabel('Matrix Size (N x N)')
plt.ylabel('Execution Time (seconds)')
plt.legend()
plt.grid(True)
plt.xticks(sizes)

output_path = os.path.join(output_dir, "comparison_plot.png")
plt.savefig(output_path)
plt.show()

print(f"Plot saved as {output_path}")
