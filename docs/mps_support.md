# Apple Silicon (MPS) Support

Flexynesis now supports GPU acceleration on Apple Silicon devices using the Metal Performance Shaders (MPS) framework. This enables GPU-accelerated workflows on modern Macs while maintaining full backward compatibility with CUDA and CPU backends.

## Device Selection

Flexynesis automatically detects and selects the best available device in the following priority order:

1. **CUDA** (if available on Linux/Windows with NVIDIA GPU)
2. **MPS** (if available on Apple Silicon Macs)  
3. **CPU** (fallback for all systems)

### Automatic Device Detection

By default, Flexynesis will automatically select the optimal device:

```bash
flexynesis --data_path your_data --model_class DirectPred --target_variables your_target
```

### Manual Device Selection

You can explicitly specify which device to use with the `--device` argument:

```bash
# Use MPS explicitly
flexynesis --data_path your_data --model_class DirectPred --target_variables your_target --device mps

# Use CUDA explicitly  
flexynesis --data_path your_data --model_class DirectPred --target_variables your_target --device cuda

# Use CPU explicitly
flexynesis --data_path your_data --model_class DirectPred --target_variables your_target --device cpu

# Automatic selection (default)
flexynesis --data_path your_data --model_class DirectPred --target_variables your_target --device auto
```

### Legacy GPU Flag

The legacy `--use_gpu` flag is still supported for backward compatibility:

```bash
# Legacy flag (will use best GPU available: CUDA > MPS)
flexynesis --data_path your_data --model_class DirectPred --target_variables your_target --use_gpu
```

## MPS Compatibility

### Automatic Tensor Conversion

MPS has some limitations compared to CUDA. Flexynesis automatically handles these differences:

- **Float64 to Float32**: MPS doesn't support float64, so tensors are automatically converted to float32
- **Safe Device Transfer**: All tensor operations use safe transfer functions to ensure MPS compatibility

### Memory Management

MPS memory management is handled automatically. You can monitor memory usage as usual, and Flexynesis will ensure optimal memory allocation patterns for Apple Silicon devices.

## System Requirements

- **macOS**: 12.3 or later
- **Hardware**: Apple Silicon (M1, M1 Pro, M1 Max, M1 Ultra, M2, M2 Pro, M2 Max, M2 Ultra, or newer)
- **Python**: 3.11 or later
- **PyTorch**: Latest version with MPS support

## Verification

To verify that MPS is available on your system, you can run:

```python
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
```

## Performance Notes

- MPS provides significant speedup over CPU for neural network training and inference
- Performance depends on model size, batch size, and available unified memory
- For very small models or datasets, CPU might be faster due to overhead
- Memory usage is more efficient compared to discrete GPUs as it uses unified memory architecture
