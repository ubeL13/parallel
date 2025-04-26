import numpy as np
import os

sizes = list(range(100, 2200, 200))

os.makedirs("matrices", exist_ok=True)

for idx, size in enumerate(sizes, 1):
    rowsA, colsA = size, size
    rowsB, colsB = size, size

    A = np.random.uniform(-10, 10, (rowsA, colsA))
    B = np.random.uniform(-10, 10, (rowsB, colsB))

    fileA = f"matrices/matrixA_{idx}.txt"
    fileB = f"matrices/matrixB_{idx}.txt"

    with open(fileA, 'w') as f:
        f.write(f"{rowsA} {colsA}\n")
        for row in A:
            f.write(' '.join(f"{val:.6f}" for val in row) + "\n")

    with open(fileB, 'w') as f:
        f.write(f"{rowsB} {colsB}\n")
        for row in B:
            f.write(' '.join(f"{val:.6f}" for val in row) + "\n")

    print(f"Generated matrices: matrixA_{idx}.txt and matrixB_{idx}.txt")

print("All matrices generated successfully.")