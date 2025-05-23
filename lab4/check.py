import numpy as np
import os
from pathlib import Path

results_dir = "lab4/results"  
matrices_dir = "matrices"
report_file = os.path.join(results_dir, "verification_report.txt")

Path(results_dir).mkdir(parents=True, exist_ok=True)
Path(matrices_dir).mkdir(parents=True, exist_ok=True)

EPSILON = 1e-4


def read_matrix(filepath):
    try:
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
            if not lines:
                raise ValueError(f"Empty file: {filepath}")
            rows, cols = map(int, lines[0].split())

            data = []
            for line in lines[1:rows+1]:
                values = list(map(float, line.split()))
                if len(values) != cols:
                    raise ValueError(f"Invalid column count in {filepath}")
                data.append(values)

            return np.array(data)
    except Exception as e:
        print(f"Error reading {filepath}: {str(e)}")
        raise


def verify_matrices():
    report_lines = []
    idx = 1
    all_passed = True

    while True:
        try:
            fileA = os.path.join(matrices_dir, f"matrixA_{idx}.txt")
            fileB = os.path.join(matrices_dir, f"matrixB_{idx}.txt")
            fileC = os.path.join(results_dir, f"matrixC_{idx}.txt")

            if not all(os.path.exists(f) for f in [fileA, fileB, fileC]):
                if idx == 1:
                    print("No test files found in matrices/ directory")
                break

            print(f"\nVerifying test case {idx}...")

            A = read_matrix(fileA)
            B = read_matrix(fileB)
            C_calc = read_matrix(fileC)

            if A.shape[1] != B.shape[0]:
                report_lines.append(
                    f"Test {idx}: FAILED - Dimension mismatch (A cols {A.shape[1]} != B rows {B.shape[0]})\n")
                all_passed = False
                idx += 1
                continue

            if C_calc.shape != (A.shape[0], B.shape[1]):
                report_lines.append(
                    f"Test {idx}: FAILED - Result matrix has wrong dimensions (expected {A.shape[0]}x{B.shape[1]}, got {C_calc.shape})\n")
                all_passed = False
                idx += 1
                continue

            C_expected = np.matmul(A, B)

            if np.allclose(C_calc, C_expected, atol=EPSILON):
                report_lines.append(f"Test {idx}: PASSED\n")
                print("PASSED")
            else:
 
                max_diff = np.max(np.abs(C_calc - C_expected))
                report_lines.append(
                    f"Test {idx}: FAILED - Max difference: {max_diff:.2e} (allowed: {EPSILON:.0e})\n")
                all_passed = False
                print(f"FAILED (max diff: {max_diff:.2e})")

            idx += 1

        except Exception as e:
            report_lines.append(f"Test {idx}: ERROR - {str(e)}\n")
            all_passed = False
            idx += 1
            print(f"! ERROR: {str(e)}")

    with open(report_file, 'w') as f:
        f.writelines(report_lines)

    if all_passed:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed. See verification_report.txt")

    return all_passed


if __name__ == "__main__":
    verify_matrices()
