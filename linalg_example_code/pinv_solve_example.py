#!/usr/bin/env python3
"""
Example of using numpy.linalg.pinv, numpy.linalg.solve, torch.linalg.solve, torch.linalg.pinv,
cupy.linalg.solve, and cupy.linalg.pinv for computing ((Y Y^T + εI)^-1) * b
for a large square matrix Y and vector b with Tikhonov regularization.

This script demonstrates:
1. Creating a large random square matrix Y (size x size) and vector b (size x 1)
2. Computing Y Y^T + epsilon * I (Tikhonov regularization for numerical stability)
3. Using numpy.linalg.pinv (CPU) to compute ((Y Y^T + εI)^-1) * b
4. Using numpy.linalg.solve (CPU) to directly solve (Y Y^T + εI) x = b
5. Using torch.linalg.pinv (GPU/CUDA) for computing ((Y Y^T + εI)^-1) * b
6. Using torch.linalg.solve (GPU/CUDA) to directly solve (Y Y^T + εI) x = b
7. Using torch.linalg.pinv (GPU/CUDA) batched for computing ((Y Y^T + εI)^-1) * b (batch=100)
8. Using torch.linalg.solve (GPU/CUDA) batched to directly solve (Y Y^T + εI) x = b (batch=100)
9. Using cupy.linalg.pinv (GPU/CUDA) for computing ((Y Y^T + εI)^-1) * b
10. Using cupy.linalg.solve (GPU/CUDA) to directly solve (Y Y^T + εI) x = b
11. Performance timing and verification comparing different methods

Usage: python pinv_solve_example.py [--size 5000] [--epsilon 0.0]
"""

import argparse
import time
from typing import Tuple

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Install with: pip install torch")

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    print(
        "Warning: CuPy not available. Install with: pip install cupy-cuda11x or cupy-cuda12x"
    )


def create_large_matrix_and_vector(
    size: int = 5000, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a large random matrix and vector for testing."""
    np.random.seed(seed)
    Y = np.random.randn(size, size)
    b = np.random.randn(size, 1)
    return Y, b


def compute_YYt_pinv_solve_method1(
    Y: np.ndarray, b: np.ndarray, epsilon: float = 0.0
) -> Tuple[np.ndarray, float]:
    """
    Method 1: Direct computation using Y Y^T + εI then pinv, then solve
    """
    start_time = time.time()

    # Compute Y Y^T + epsilon * I (Tikhonov regularization)
    YYt = Y @ Y.T + epsilon * np.eye(Y.shape[0])

    # Compute pseudo-inverse of Y Y^T + εI
    YYt_pinv = np.linalg.pinv(YYt)

    # Solve: (Y Y^T + εI)^-1 * b
    result = YYt_pinv @ b

    elapsed_time = time.time() - start_time
    return result, elapsed_time


def compute_YYt_solve_method2(
    Y: np.ndarray, b: np.ndarray, epsilon: float = 0.0
) -> Tuple[np.ndarray, float]:
    """
    Method 2: Using numpy.linalg.solve to directly solve (Y Y^T + εI) x = b
    This is more efficient and numerically stable than computing pseudo-inverse
    """
    start_time = time.time()

    # Compute Y Y^T + epsilon * I (Tikhonov regularization)
    YYt = Y @ Y.T + epsilon * np.eye(Y.shape[0])

    # Solve: (Y Y^T + εI) x = b directly using linalg.solve
    result = np.linalg.solve(YYt, b)

    elapsed_time = time.time() - start_time
    return result, elapsed_time


def compute_YYt_solve_torch_gpu(
    Y: np.ndarray, b: np.ndarray, epsilon: float = 0.0
) -> Tuple[np.ndarray, float, bool]:
    """
    Method 3: Using PyTorch with GPU acceleration for solving (Y Y^T + εI) x = b directly
    Note: Only measures GPU computation time, excluding data transfer overhead
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is not available. Please install PyTorch.")

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_used = torch.cuda.is_available()

    # Convert to PyTorch tensor and move to GPU (not timed)
    Y_torch = torch.from_numpy(Y.astype(np.float32)).to(device)
    b_torch = torch.from_numpy(b.astype(np.float32)).to(device)

    # Synchronize GPU before timing (ensures all previous operations are complete)
    if gpu_used:
        torch.cuda.synchronize()

    # Start timing - only measure GPU computation
    start_time = time.time()

    # Compute Y Y^T + epsilon * I (Tikhonov regularization)
    YYt_torch = Y_torch @ Y_torch.T + epsilon * torch.eye(
        Y_torch.shape[0], device=device, dtype=Y_torch.dtype
    )

    # Solve: (Y Y^T + εI) x = b directly using torch.linalg.solve
    result_torch = torch.linalg.solve(YYt_torch, b_torch)

    # Synchronize GPU to ensure computation is complete before stopping timer
    if gpu_used:
        torch.cuda.synchronize()

    # Stop timing
    elapsed_time = time.time() - start_time

    # Convert back to numpy (not timed)
    result = result_torch.cpu().numpy().astype(np.float64)

    return result, elapsed_time, gpu_used


def compute_YYt_pinv_solve_torch_gpu(
    Y: np.ndarray, b: np.ndarray, epsilon: float = 0.0
) -> Tuple[np.ndarray, float, bool]:
    """
    Method 4: Using PyTorch with GPU acceleration for computing ((Y Y^T + εI)^-1) * b
    Note: Only measures GPU computation time, excluding data transfer overhead
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is not available. Please install PyTorch.")

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_used = torch.cuda.is_available()

    # Convert to PyTorch tensor and move to GPU (not timed)
    Y_torch = torch.from_numpy(Y.astype(np.float32)).to(device)
    b_torch = torch.from_numpy(b.astype(np.float32)).to(device)

    # Synchronize GPU before timing (ensures all previous operations are complete)
    if gpu_used:
        torch.cuda.synchronize()

    # Start timing - only measure GPU computation
    start_time = time.time()

    # Compute Y Y^T + epsilon * I (Tikhonov regularization)
    YYt_torch = Y_torch @ Y_torch.T + epsilon * torch.eye(
        Y_torch.shape[0], device=device, dtype=Y_torch.dtype
    )

    # Compute pseudo-inverse using PyTorch
    YYt_pinv_torch = torch.linalg.pinv(YYt_torch)

    # Solve: (Y Y^T + εI)^-1 * b
    result_torch = YYt_pinv_torch @ b_torch

    # Synchronize GPU to ensure computation is complete before stopping timer
    if gpu_used:
        torch.cuda.synchronize()

    # Stop timing
    elapsed_time = time.time() - start_time

    # Convert back to numpy (not timed)
    result = result_torch.cpu().numpy().astype(np.float64)

    return result, elapsed_time, gpu_used


def compute_YYt_pinv_solve_torch_batched(
    Y: np.ndarray, b: np.ndarray, epsilon: float = 0.0, batch_size: int = 100
) -> Tuple[np.ndarray, float, bool]:
    """
    Method 5: Using PyTorch with GPU acceleration for batched computing ((Y Y^T + εI)^-1) * b
    Note: Only measures GPU computation time, excluding data transfer overhead
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is not available. Please install PyTorch.")

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_used = torch.cuda.is_available()

    # Convert to PyTorch tensor and move to GPU (not timed)
    Y_torch = torch.from_numpy(Y.astype(np.float32)).to(device)
    b_torch = torch.from_numpy(b.astype(np.float32)).to(device)

    # Create batched data by repeating the same problem
    Y_batched = Y_torch.unsqueeze(0).repeat(
        batch_size, 1, 1
    )  # (batch_size, rows, cols)
    b_batched = b_torch.unsqueeze(0).repeat(batch_size, 1, 1)  # (batch_size, rows, 1)

    # Synchronize GPU before timing
    if gpu_used:
        torch.cuda.synchronize()

    # Start timing - only measure GPU computation
    start_time = time.time()

    # Compute batched Y Y^T + epsilon * I
    YYt_batched = torch.bmm(
        Y_batched, Y_batched.transpose(-2, -1)
    )  # Batched matrix multiplication
    if epsilon > 0:
        eye_batch = epsilon * torch.eye(
            Y_batched.shape[1], device=device, dtype=Y_batched.dtype
        ).unsqueeze(0).repeat(batch_size, 1, 1)
        YYt_batched = YYt_batched + eye_batch

    # Compute batched pseudo-inverse
    YYt_pinv_batched = torch.linalg.pinv(YYt_batched)

    # Solve batched: (Y Y^T + εI)^-1 * b
    result_batched = torch.bmm(YYt_pinv_batched, b_batched)

    # Synchronize GPU to ensure computation is complete
    if gpu_used:
        torch.cuda.synchronize()

    # Stop timing
    elapsed_time = time.time() - start_time
    avg_time = elapsed_time / batch_size  # Average time per batch item

    # Convert back to numpy (not timed) - take first result as representative
    result = result_batched[0].cpu().numpy().astype(np.float64)

    return result, avg_time, gpu_used


def compute_YYt_solve_torch_batched(
    Y: np.ndarray, b: np.ndarray, epsilon: float = 0.0, batch_size: int = 100
) -> Tuple[np.ndarray, float, bool]:
    """
    Method 6: Using PyTorch with GPU acceleration for batched solving (Y Y^T + εI) x = b
    Note: Only measures GPU computation time, excluding data transfer overhead
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is not available. Please install PyTorch.")

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_used = torch.cuda.is_available()

    # Convert to PyTorch tensor and move to GPU (not timed)
    Y_torch = torch.from_numpy(Y.astype(np.float32)).to(device)
    b_torch = torch.from_numpy(b.astype(np.float32)).to(device)

    # Create batched data by repeating the same problem
    Y_batched = Y_torch.unsqueeze(0).repeat(
        batch_size, 1, 1
    )  # (batch_size, rows, cols)
    b_batched = b_torch.unsqueeze(0).repeat(batch_size, 1, 1)  # (batch_size, rows, 1)

    # Synchronize GPU before timing
    if gpu_used:
        torch.cuda.synchronize()

    # Start timing - only measure GPU computation
    start_time = time.time()

    # Compute batched Y Y^T + epsilon * I
    YYt_batched = torch.bmm(
        Y_batched, Y_batched.transpose(-2, -1)
    )  # Batched matrix multiplication
    if epsilon > 0:
        eye_batch = epsilon * torch.eye(
            Y_batched.shape[1], device=device, dtype=Y_batched.dtype
        ).unsqueeze(0).repeat(batch_size, 1, 1)
        YYt_batched = YYt_batched + eye_batch

    # Solve batched: (Y Y^T + εI) x = b using batched solve
    result_batched = torch.linalg.solve(YYt_batched, b_batched)

    # Synchronize GPU to ensure computation is complete
    if gpu_used:
        torch.cuda.synchronize()

    # Stop timing
    elapsed_time = time.time() - start_time
    avg_time = elapsed_time / batch_size  # Average time per batch item

    # Convert back to numpy (not timed) - take first result as representative
    result = result_batched[0].cpu().numpy().astype(np.float64)

    return result, avg_time, gpu_used


def compute_YYt_solve_cupy_gpu(
    Y: np.ndarray, b: np.ndarray, epsilon: float = 0.0
) -> Tuple[np.ndarray, float, bool]:
    """
    Method 5: Using CuPy with GPU acceleration for solving (Y Y^T + εI) x = b directly
    Note: Only measures GPU computation time, excluding data transfer overhead
    """
    if not CUPY_AVAILABLE:
        raise ImportError("CuPy is not available. Please install CuPy.")

    gpu_used = True  # CuPy always uses GPU

    # Convert to CuPy array and move to GPU (not timed)
    Y_cupy = cp.asarray(Y.astype(np.float32))
    b_cupy = cp.asarray(b.astype(np.float32))

    # Synchronize GPU before timing
    cp.cuda.Stream.null.synchronize()

    # Start timing - only measure GPU computation
    start_time = time.time()

    # Compute Y Y^T + epsilon * I (Tikhonov regularization)
    YYt_cupy = Y_cupy @ Y_cupy.T + epsilon * cp.eye(Y_cupy.shape[0], dtype=Y_cupy.dtype)

    # Solve: (Y Y^T + εI) x = b directly using cp.linalg.solve
    result_cupy = cp.linalg.solve(YYt_cupy, b_cupy)

    # Synchronize GPU to ensure computation is complete before stopping timer
    cp.cuda.Stream.null.synchronize()

    # Stop timing
    elapsed_time = time.time() - start_time

    # Convert back to numpy (not timed)
    result = cp.asnumpy(result_cupy).astype(np.float64)

    return result, elapsed_time, gpu_used


def compute_YYt_pinv_solve_cupy_gpu(
    Y: np.ndarray, b: np.ndarray, epsilon: float = 0.0
) -> Tuple[np.ndarray, float, bool]:
    """
    Method 6: Using CuPy with GPU acceleration for computing ((Y Y^T + εI)^-1) * b
    Note: Only measures GPU computation time, excluding data transfer overhead
    """
    if not CUPY_AVAILABLE:
        raise ImportError("CuPy is not available. Please install CuPy.")

    gpu_used = True  # CuPy always uses GPU

    # Convert to CuPy array and move to GPU (not timed)
    Y_cupy = cp.asarray(Y.astype(np.float32))
    b_cupy = cp.asarray(b.astype(np.float32))

    # Synchronize GPU before timing
    cp.cuda.Stream.null.synchronize()

    # Start timing - only measure GPU computation
    start_time = time.time()

    # Compute Y Y^T + epsilon * I (Tikhonov regularization)
    YYt_cupy = Y_cupy @ Y_cupy.T + epsilon * cp.eye(Y_cupy.shape[0], dtype=Y_cupy.dtype)

    # Compute pseudo-inverse using CuPy
    YYt_pinv_cupy = cp.linalg.pinv(YYt_cupy)

    # Solve: (Y Y^T + εI)^-1 * b
    result_cupy = YYt_pinv_cupy @ b_cupy

    # Synchronize GPU to ensure computation is complete before stopping timer
    cp.cuda.Stream.null.synchronize()

    # Stop timing
    elapsed_time = time.time() - start_time

    # Convert back to numpy (not timed)
    result = cp.asnumpy(result_cupy).astype(np.float64)

    return result, elapsed_time, gpu_used


def verify_solution(
    Y: np.ndarray,
    b: np.ndarray,
    result: np.ndarray,
    epsilon: float = 0.0,
    tolerance: float = 1e-4,
) -> Tuple[bool, float]:
    """
    Verify the solution by comparing with numpy's lstsq solver.
    """
    # Use numpy's least squares solver as reference with regularization
    YYt_reg = Y @ Y.T + epsilon * np.eye(Y.shape[0])
    reference_solution, _, _, _ = np.linalg.lstsq(YYt_reg, b, rcond=None)

    # Calculate the difference
    diff = np.linalg.norm(result - reference_solution)

    return diff < tolerance, diff


def check_system_capabilities():
    """Check and display system capabilities for PyTorch, CuPy and CUDA."""
    print("System Capabilities:")
    print("-" * 20)
    print(f"NumPy version: {np.__version__}")
    if TORCH_AVAILABLE:
        print(f"PyTorch version: {torch.__version__}")
        print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU device: {torch.cuda.get_device_name(0)}")
            print(
                f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
            )
    else:
        print("PyTorch: Not available")

    if CUPY_AVAILABLE:
        print(f"CuPy version: {cp.__version__}")
        print(f"CuPy CUDA available: {cp.cuda.is_available()}")
        if cp.cuda.is_available():
            print(
                f"CuPy GPU device: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}"
            )
    else:
        print("CuPy: Not available")
    print()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Compare linear solvers with Tikhonov regularization"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="Regularization parameter (default: 0.0)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=5000,
        help="Matrix size (creates size x size matrix, default: 5000)",
    )
    args = parser.parse_args()

    print("Complete Comparison: NumPy vs PyTorch vs CuPy (solve/pinv methods)")
    print("=" * 70)
    print(f"Matrix size: {args.size} x {args.size}")
    print(f"Using regularization parameter epsilon = {args.epsilon}")
    print()

    # Check system capabilities
    check_system_capabilities()

    # Create a large random matrix Y and vector b
    print(
        f"Creating {args.size}x{args.size} random matrix Y and {args.size}x1 vector b..."
    )
    Y, b = create_large_matrix_and_vector(args.size)
    print(f"Matrix Y shape: {Y.shape}")
    print(f"Vector b shape: {b.shape}")
    print(f"Matrix Y dtype: {Y.dtype}")
    print(f"Vector b dtype: {b.dtype}")
    print(f"Matrix Y memory usage: {Y.nbytes / 1024**2:.2f} MB")
    print(f"Vector b memory usage: {b.nbytes / 1024:.2f} KB")
    print()

    # Compute Y Y^T first (for verification)
    print("Computing Y Y^T...")
    start_time = time.time()
    YYt = Y @ Y.T
    YYt_time = time.time() - start_time
    print(f"Y Y^T computed in {YYt_time:.4f} seconds")
    print(f"Y Y^T shape: {YYt.shape}")
    print()

    # Method 1: Direct computation using NumPy pinv
    print("Method 1: Computing ((Y Y^T + εI)^-1) * b using NumPy pinv...")
    result1, time1 = compute_YYt_pinv_solve_method1(Y, b, args.epsilon)
    print(f"Method 1 completed in {time1:.4f} seconds")
    print(f"Result shape: {result1.shape}")

    # Verify the result
    is_valid1, error1 = verify_solution(Y, b, result1, args.epsilon)
    print(f"Verification: {'PASSED' if is_valid1 else 'FAILED'}")
    print(f"Error vs reference solution: {error1:.2e}")
    print()

    # Method 2: Using NumPy solve
    print("Method 2: Solving (Y Y^T + εI) x = b using NumPy solve...")
    result2, time2 = compute_YYt_solve_method2(Y, b, args.epsilon)
    print(f"Method 2 completed in {time2:.4f} seconds")
    print(f"Result shape: {result2.shape}")

    # Verify the result
    is_valid2, error2 = verify_solution(Y, b, result2, args.epsilon)
    print(f"Verification: {'PASSED' if is_valid2 else 'FAILED'}")
    print(f"Error vs reference solution: {error2:.2e}")
    print()

    # Initialize variables for methods that might be skipped
    time3, gpu_used3 = None, False
    time4, gpu_used4 = None, False
    time5, gpu_used5 = None, False
    time6, gpu_used6 = None, False
    time7, gpu_used7 = None, False
    time8, gpu_used8 = None, False

    # Method 3: Using PyTorch GPU pinv approach (if available)
    if TORCH_AVAILABLE:
        print("Method 3: Computing ((Y Y^T + εI)^-1) * b using PyTorch with GPU...")
        result3, time3, gpu_used3 = compute_YYt_pinv_solve_torch_gpu(Y, b, args.epsilon)
        print(f"Method 3 completed in {time3:.4f} seconds")
        print(f"Device used: {'GPU (CUDA)' if gpu_used3 else 'CPU'}")
        print(f"Result shape: {result3.shape}")

        # Verify the result
        is_valid3, error3 = verify_solution(Y, b, result3, args.epsilon)
        print(f"Verification: {'PASSED' if is_valid3 else 'FAILED'}")
        print(f"Error vs reference solution: {error3:.2e}")
        print()
    else:
        print("Method 3: Skipped (PyTorch not available)")
        print("Install PyTorch with: pip install torch")
        print()

    # Method 4: Using PyTorch GPU solve approach (if available)
    if TORCH_AVAILABLE:
        print("Method 4: Solving (Y Y^T + εI) x = b using PyTorch with GPU...")
        result4, time4, gpu_used4 = compute_YYt_solve_torch_gpu(Y, b, args.epsilon)
        print(f"Method 4 completed in {time4:.4f} seconds")
        print(f"Device used: {'GPU (CUDA)' if gpu_used4 else 'CPU'}")
        print(f"Result shape: {result4.shape}")

        # Verify the result
        is_valid4, error4 = verify_solution(Y, b, result4, args.epsilon)
        print(f"Verification: {'PASSED' if is_valid4 else 'FAILED'}")
        print(f"Error vs reference solution: {error4:.2e}")
        print()
    else:
        print("Method 4: Skipped (PyTorch not available)")
        print("Install PyTorch with: pip install torch")
        print()

    # Method 5: Using PyTorch GPU batched pinv approach (if available)
    if TORCH_AVAILABLE:
        print(
            "Method 5: Computing ((Y Y^T + εI)^-1) * b using PyTorch batched pinv (batch=100)..."
        )
        result5, time5, gpu_used5 = compute_YYt_pinv_solve_torch_batched(
            Y, b, args.epsilon, batch_size=100
        )
        print(f"Method 5 completed in {time5:.4f} seconds (avg per batch item)")
        print(f"Device used: {'GPU (CUDA)' if gpu_used5 else 'CPU'}")
        print(f"Result shape: {result5.shape}")

        # Verify the result
        is_valid5, error5 = verify_solution(Y, b, result5, args.epsilon)
        print(f"Verification: {'PASSED' if is_valid5 else 'FAILED'}")
        print(f"Error vs reference solution: {error5:.2e}")
        print()
    else:
        print("Method 5: Skipped (PyTorch not available)")
        print("Install PyTorch with: pip install torch")
        print()

    # Method 6: Using PyTorch GPU batched solve approach (if available)
    if TORCH_AVAILABLE:
        print(
            "Method 6: Solving (Y Y^T + εI) x = b using PyTorch batched solve (batch=100)..."
        )
        result6, time6, gpu_used6 = compute_YYt_solve_torch_batched(
            Y, b, args.epsilon, batch_size=100
        )
        print(f"Method 6 completed in {time6:.4f} seconds (avg per batch item)")
        print(f"Device used: {'GPU (CUDA)' if gpu_used6 else 'CPU'}")
        print(f"Result shape: {result6.shape}")

        # Verify the result
        is_valid6, error6 = verify_solution(Y, b, result6, args.epsilon)
        print(f"Verification: {'PASSED' if is_valid6 else 'FAILED'}")
        print(f"Error vs reference solution: {error6:.2e}")
        print()
    else:
        print("Method 6: Skipped (PyTorch not available)")
        print("Install PyTorch with: pip install torch")
        print()

    # Method 7: Using CuPy GPU pinv approach (if available)
    if CUPY_AVAILABLE:
        print("Method 7: Computing ((Y Y^T + εI)^-1) * b using CuPy with GPU...")
        result7, time7, gpu_used7 = compute_YYt_pinv_solve_cupy_gpu(Y, b, args.epsilon)
        print(f"Method 7 completed in {time7:.4f} seconds")
        print(f"Device used: GPU (CUDA)")
        print(f"Result shape: {result7.shape}")

        # Verify the result
        is_valid7, error7 = verify_solution(Y, b, result7, args.epsilon)
        print(f"Verification: {'PASSED' if is_valid7 else 'FAILED'}")
        print(f"Error vs reference solution: {error7:.2e}")
        print()
    else:
        print("Method 7: Skipped (CuPy not available)")
        print("Install CuPy with: pip install cupy-cuda11x or cupy-cuda12x")
        print()

    # Method 8: Using CuPy GPU solve approach (if available)
    if CUPY_AVAILABLE:
        print("Method 8: Solving (Y Y^T + εI) x = b using CuPy with GPU...")
        result8, time8, gpu_used8 = compute_YYt_solve_cupy_gpu(Y, b, args.epsilon)
        print(f"Method 8 completed in {time8:.4f} seconds")
        print(f"Device used: GPU (CUDA)")
        print(f"Result shape: {result8.shape}")

        # Verify the result
        is_valid8, error8 = verify_solution(Y, b, result8, args.epsilon)
        print(f"Verification: {'PASSED' if is_valid8 else 'FAILED'}")
        print(f"Error vs reference solution: {error8:.2e}")
        print()
    else:
        print("Method 8: Skipped (CuPy not available)")
        print("Install CuPy with: pip install cupy-cuda11x or cupy-cuda12x")
        print()

    # Performance comparison summary
    print("Performance Comparison Summary:")
    print("=" * 55)
    print("CPU Methods:")
    print(f"Method 1 (NumPy pinv):             {time1:.4f} seconds")
    print(f"Method 2 (NumPy solve):            {time2:.4f} seconds")
    print(f"NumPy solve speedup vs pinv:       {time1/time2:.2f}x")
    print()

    if TORCH_AVAILABLE and time3 is not None and time4 is not None:
        print("PyTorch GPU Methods:")
        print(f"Method 3 (PyTorch pinv):           {time3:.4f} seconds")
        print(f"Method 4 (PyTorch solve):          {time4:.4f} seconds")
        print(f"PyTorch pinv vs solve ratio:       {time4/time3:.2f}x")
        print()

    if TORCH_AVAILABLE and time5 is not None and time6 is not None:
        print("PyTorch GPU Batched Methods:")
        print(f"Method 5 (PyTorch batched pinv):   {time5:.4f} seconds (avg per item)")
        print(f"Method 6 (PyTorch batched solve):  {time6:.4f} seconds (avg per item)")
        print(f"Batched pinv vs solve ratio:       {time6/time5:.2f}x")
        print()

    if CUPY_AVAILABLE and time7 is not None and time8 is not None:
        print("CuPy GPU Methods:")
        print(f"Method 7 (CuPy pinv):              {time7:.4f} seconds")
        print(f"Method 8 (CuPy solve):             {time8:.4f} seconds")
        print(f"CuPy pinv vs solve ratio:          {time8/time7:.2f}x")
        print()

    # Comprehensive comparison table
    print("=" * 80)
    print("COMPREHENSIVE PERFORMANCE AND ACCURACY COMPARISON")
    print("=" * 80)

    # Collect all available results
    methods_data = []

    # Method 1 (always available)
    methods_data.append(
        {
            "method": "Method 1 (NumPy pinv)",
            "time": time1,
            "speedup": 1.0,  # Reference method
            "error": error1,
            "status": "PASSED" if is_valid1 else "FAILED",
        }
    )

    # Method 2 (always available)
    methods_data.append(
        {
            "method": "Method 2 (NumPy solve)",
            "time": time2,
            "speedup": time1 / time2 if time2 > 0 else 0,
            "error": error2,
            "status": "PASSED" if is_valid2 else "FAILED",
        }
    )

    # Method 3 (PyTorch pinv)
    if TORCH_AVAILABLE and time3 is not None:
        methods_data.append(
            {
                "method": "Method 3 (PyTorch pinv)",
                "time": time3,
                "speedup": time1 / time3 if time3 > 0 else 0,
                "error": error3,
                "status": "PASSED" if is_valid3 else "FAILED",
            }
        )

    # Method 4 (PyTorch solve)
    if TORCH_AVAILABLE and time4 is not None:
        methods_data.append(
            {
                "method": "Method 4 (PyTorch solve)",
                "time": time4,
                "speedup": time1 / time4 if time4 > 0 else 0,
                "error": error4,
                "status": "PASSED" if is_valid4 else "FAILED",
            }
        )

    # Method 5 (PyTorch batched pinv)
    if TORCH_AVAILABLE and time5 is not None:
        methods_data.append(
            {
                "method": "Method 5 (PT batch pinv)",
                "time": time5,
                "speedup": time1 / time5 if time5 > 0 else 0,
                "error": error5,
                "status": "PASSED" if is_valid5 else "FAILED",
            }
        )

    # Method 6 (PyTorch batched solve)
    if TORCH_AVAILABLE and time6 is not None:
        methods_data.append(
            {
                "method": "Method 6 (PT batch solve)",
                "time": time6,
                "speedup": time1 / time6 if time6 > 0 else 0,
                "error": error6,
                "status": "PASSED" if is_valid6 else "FAILED",
            }
        )

    # Method 7 (CuPy pinv)
    if CUPY_AVAILABLE and time7 is not None:
        methods_data.append(
            {
                "method": "Method 7 (CuPy pinv)",
                "time": time7,
                "speedup": time1 / time7 if time7 > 0 else 0,
                "error": error7,
                "status": "PASSED" if is_valid7 else "FAILED",
            }
        )

    # Method 8 (CuPy solve)
    if CUPY_AVAILABLE and time8 is not None:
        methods_data.append(
            {
                "method": "Method 8 (CuPy solve)",
                "time": time8,
                "speedup": time1 / time8 if time8 > 0 else 0,
                "error": error8,
                "status": "PASSED" if is_valid8 else "FAILED",
            }
        )

    # Print table header
    print(
        f"{'Method':<25} {'Time (s)':<10} {'Speedup vs M1':<12} {'Error':<12} {'Status':<8}"
    )
    print("-" * 80)

    # Print table rows
    for data in methods_data:
        print(
            f"{data['method']:<25} {data['time']:<10.4f} {data['speedup']:<12.2f} {data['error']:<12.2e} {data['status']:<8}"
        )

    print("-" * 80)


if __name__ == "__main__":
    main()
