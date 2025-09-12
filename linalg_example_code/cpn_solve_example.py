import argparse
import os
import sys

import cupy as cp
import cupynumeric as np
import torch
from legate.timing import time

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Test CuPy-Numeric linear solver with optional CuPy comparison"
)
parser.add_argument(
    "--size",
    type=int,
    default=1000,
    help="Matrix size (creates size x size matrix, default: 1000)",
)
parser.add_argument(
    "--skip-cupy",
    action="store_true",
    help="Skip CuPy comparison (only run CuPy-Numeric)",
)
args = parser.parse_args()

size = args.size
print(f"Matrix size: {size} x {size}")
print(f"Memory usage per matrix: {(size*size*4)/(1024**2):.2f} MB (float32)")
print()

print("\n" + "=" * 50)
print("CuPy-Numeric Linear System Solver Test")
print("=" * 50)

# Solve Y^T Y x = b using cupynumeric.linalg.solve
print(f"Creating {size}x{size} matrix Y and {size}x1 vector b...")
start_time = time()
Y = np.random.randn(size, size).astype(np.float32)
b = np.random.randn(size, 1).astype(np.float32)

# Compute Y Y^T
print("Computing Y Y^T...")
YtY = Y @ Y.T

# Solve Y^T Y x = b
print("Solving Y Y^T x = b using cupynumeric.linalg.solve...")
x = np.linalg.solve(YtY, b)

end_time = time()
elapsed_time = (end_time - start_time) / 1000

print(f"Problem completed!")
print(f"Matrix Y shape: {Y.shape}")
print(f"Vector b shape: {b.shape}")
print(f"Solution x shape: {x.shape}")
print(f"Total time (Y^T Y computation + solve): {elapsed_time:.4f} ms")

# Verify solution
print("\nVerification:")
residual = YtY @ x - b
residual_norm = np.linalg.norm(residual)
print(f"Residual norm ||Y^T Y x - b||: {residual_norm:.2e}")
print("Verification: PASSED" if residual_norm < 1e-3 else "Verification: FAILED")

# example code to convert cupynumeric -> cupy -> pytorch
x_cupy = cp.asarray(x)
x_torch = torch.as_tensor(x_cupy, device="cuda")

if not args.skip_cupy:
    print("\n" + "=" * 50)
    print("CuPy Linear System Solver Test (Comparison)")
    print("=" * 50)

    # Solve Y^T Y x = b using cupy.linalg.solve for comparison
    print(f"Creating {size}x{size} matrix Y and {size}x1 vector b...")
    start_time = time()
    Y_cp = cp.random.randn(size, size).astype(cp.float32)
    b_cp = cp.random.randn(size, 1).astype(cp.float32)

    # Compute Y^T Y
    print("Computing Y Y^T...")
    YtY_cp = Y_cp @ Y_cp.T

    # Solve Y^T Y x = b
    print("Solving Y Y^T x = b using cupy.linalg.solve...")
    x_cp = cp.linalg.solve(YtY_cp, b_cp)

    # Synchronize GPU to ensure completion
    cp.cuda.Stream.null.synchronize()

    end_time = time()
    elapsed_time_cupy = (end_time - start_time) / 1000

    print(f"Problem completed!")
    print(f"Matrix Y shape: {Y_cp.shape}")
    print(f"Vector b shape: {b_cp.shape}")
    print(f"Solution x shape: {x_cp.shape}")
    print(f"Total time (Y Y^T computation + solve): {elapsed_time_cupy:.4f} ms")

    # Verify solution
    print("\nVerification:")
    residual_cp = YtY_cp @ x_cp - b_cp
    residual_norm_cp = cp.linalg.norm(residual_cp)
    print(f"Residual norm ||Y Y^T x - b||: {residual_norm_cp:.2e}")
    print("Verification: PASSED" if residual_norm_cp < 1e-3 else "Verification: FAILED")

    print("\n" + "=" * 50)
    print("Performance Comparison")
    print("=" * 50)
    print(f"CuPy-Numeric time:  {elapsed_time:.4f} ms")
    print(f"CuPy time:          {elapsed_time_cupy:.4f} ms")
    if elapsed_time_cupy > 0:
        speedup = elapsed_time_cupy / elapsed_time
        print(
            f"CuPy-Numeric vs CuPy: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}"
        )
    else:
        print("Could not compute speedup (division by zero)")
else:
    print("\n" + "=" * 50)
    print("CuPy comparison skipped (--skip-cupy flag used)")
    print("=" * 50)
    print(f"CuPy-Numeric completed in: {elapsed_time:.4f} ms")
    print("Run without --skip-cupy to compare with CuPy performance")
