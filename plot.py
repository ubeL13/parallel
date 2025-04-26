import os
import re
import matplotlib.pyplot as plt

folder = "results"

sizes = []  
times = [] 

idx = 1
while True:
    fileTime = os.path.join(folder, f"time_{idx}.txt")
    if not os.path.exists(fileTime):
        break

    with open(fileTime, 'r') as f:
        content = f.read()

        match_size = re.search(r"Matrix A: (\d+)x(\d+)", content)
        match_time = re.search(r"Execution Time \(s\): ([\d\.]+)", content)

        if match_size and match_time:
            rowsA = int(match_size.group(1))
            exec_time = float(match_time.group(1))

            sizes.append(rowsA)
            times.append(exec_time)

    idx += 1

plt.figure(figsize=(8, 6))
plt.plot(sizes, times, marker='o')
plt.title('Matrix Size vs Execution Time')
plt.xlabel('Matrix Size (n x n)')
plt.ylabel('Execution Time (seconds)')
plt.grid(True)
plt.savefig(os.path.join(folder, "size_vs_time_plot.png"))
plt.show()

print("Graph plotted and saved as size_vs_time_plot.png")
