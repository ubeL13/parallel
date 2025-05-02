import numpy as np
import os

results_dir = "lab2/results"
matrices_dir = "matrices"
report_file = os.path.join(results_dir, "verification_report.txt")

EPSILON = 1e-4

def read_matrix(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        rows, cols = map(int, lines[0].split())
        data = [[float(val) for val in line.strip().split()] for line in lines[1:]]
        return np.array(data)

report_lines = []
idx = 1
all_passed = True

while True:
    fileA = os.path.join(matrices_dir, f"matrixA_{idx}.txt")
    fileB = os.path.join(matrices_dir, f"matrixB_{idx}.txt")
    fileC = os.path.join(results_dir, f"matrixC_{idx}.txt")

    if not os.path.exists(fileA) or not os.path.exists(fileB) or not os.path.exists(fileC):
        break

    A = read_matrix(fileA)
    B = read_matrix(fileB)
    C_calc = read_matrix(fileC)
    C_expected = A @ B

    if np.allclose(C_calc, C_expected, atol=EPSILON):
        report_lines.append(f"Test {idx}: PASSED\n")
    else:
        report_lines.append(f"Test {idx}: FAILED\n")
        all_passed = False

    idx += 1

with open(report_file, 'w') as f:
    f.writelines(report_lines)

if all_passed:
    print("All tests passed!")
else:
    print("Some tests failed. See verification_report.txt")
