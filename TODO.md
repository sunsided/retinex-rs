# Future Extensions

## Optimization-Based Intrinsic Decomposition

Reference implementation in C++ uses gradient-based classification and sparse linear algebra:
- Classifies gradients: large → reflectance (edges), small → shading (smooth)
- Solves least-squares problem using sparse Cholesky decomposition
- Enforces physical constraint: reflectance ≤ 1 (log-reflectance ≤ 0)
- Grayscale-first processing naturally preserves color without MSRCR

### Key Differences from Current Gaussian Approach

| Aspect | Current (Gaussian) | C++ (Optimization) |
|--------|-------------------|-------------------|
| Speed | Fast | Slow (matrix solve) |
| Quality | Good | Better separation |
| Color handling | MSRCR restoration | Grayscale-first |
| Physical constraints | None | Reflectance ≤ 1 |
| Implementation | Simple | Complex (Eigen) |

### Implementation Notes

The C++ approach builds a sparse matrix A where:
- Rows represent gradient constraints (dx, dy) and reconstruction constraint
- Columns represent reflectance and shading pixels
- Solves `A'Ax = A'b` using Cholesky factorization

Threshold parameter controls gradient classification:
- `|gradient| > threshold` → reflectance
- `|gradient| ≤ threshold` → shading

### Potential Rust Implementation

Would require:
- Sparse matrix library (e.g., `nalgebra-sparse` or `sprs`)
- Cholesky solver for sparse systems
- Gradient computation (Sobel or central differences)
- Significant performance testing

## Other Future Extensions

- GPU acceleration (CUDA/OpenCL)
- Adaptive sigma selection based on image statistics
- Additional color restoration methods
- Batch processing for video
- WebAssembly build for browser usage
