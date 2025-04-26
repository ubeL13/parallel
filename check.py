import numpy as np
import os

# Папка с матрицами
folder = "matrices"

# Порог для сравнения чисел с плавающей точкой
EPSILON = 1e-4

def read_matrix(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        size = lines[0].strip().split()
        rows, cols = int(size[0]), int(size[1])
        matrix = []
        for line in lines[1:]:
            matrix.append([float(x) for x in line.strip().split()])
        return np.array(matrix)

idx = 1
all_passed = True

report_lines = []

while True:
    fileA = os.path.join(folder, f"matrixA_{idx}.txt")
    fileB = os.path.join(folder, f"matrixB_{idx}.txt")
    fileC = os.path.join(folder, f"matrixC_{idx}.txt")

    if not os.path.exists(fileA) or not os.path.exists(fileB) or not os.path.exists(fileC):
        break

    A = read_matrix(fileA)
    B = read_matrix(fileB)
    C_cpp = read_matrix(fileC)

    C_ref = A @ B

    if np.allclose(C_cpp, C_ref, atol=EPSILON):
        print(f"Test {idx}: PASSED")
        report_lines.append(f"Test {idx}: PASSED\n")
    else:
        print(f"Test {idx}: FAILED")
        report_lines.append(f"Test {idx}: FAILED\n")
        all_passed = False

    idx += 1

# Записываем отчёт в файл
report_file = os.path.join(folder, "verification_report.txt")
with open(report_file, "w") as f:
    f.writelines(report_lines)

if all_passed:
    print("\nAll tests passed successfully!")
else:
    print("\nSome tests failed. Check your multiplication implementation.")
