# Apple Silicon (MPS) Support

Flexynesis supports GPU acceleration on Apple Silicon devices using the Metal Performance Shaders (MPS) framework. This enables GPU-accelerated workflows on modern Macs while maintaining full backward compatibility with CUDA and CPU backends.

## Device Selection

Flexynesis automatically selects the best available device in this priority order: CUDA → MPS → CPU. You can override this with the `--device` argument:

```bash
# Automatic selection (default)
flexynesis --data_path your_data --model_class DirectPred --target_variables your_target

# Use MPS explicitly
flexynesis --data_path your_data --model_class DirectPred --target_variables your_target --device mps

# Use CUDA/CPU explicitly
flexynesis --data_path your_data --model_class DirectPred --target_variables your_target --device cuda
flexynesis --data_path your_data --model_class DirectPred --target_variables your_target --device cpu
```

**Requirements**: macOS 12.3+, Apple Silicon (M1+), Python 3.11+
