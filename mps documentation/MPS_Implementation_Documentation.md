# Implementing Apple Silicon MPS Support in Flexynesis Framework

## Executive Summary

This document explains how Flexynesis now supports GPU acceleration via Apple Silicon (MPS), in addition to its existing NVIDIA CUDA and CPU support—enabling high-performance multi-omics analysis.

---

## 1. Project Overview

### 1.1 Background

Flexynesis is a flexible deep learning toolkit for interpretable multi-omics integration and clinical outcome prediction, powered by PyTorch, PyTorch Lightning, and PyG. Originally designed for CUDA GPU acceleration, it lacked support for Apple Silicon's MPS acceleration, limiting its accessibility to researchers using Apple Silicon devices. While large-scale runs are typically performed on servers with CUDA GPU clusters, MPS support is especially useful for demos and easy local experimentation on Apple Silicon hardware.

### 1.2 Objectives

- **Primary Goal**: Add MPS support to enable GPU acceleration on Apple Silicon
- **Secondary Goals**: 
  - Maintain backward compatibility with CUDA and CPU
  - Implement automatic device detection
  - Ensure seamless user experience
  - Validate performance with real multi-omics datasets

### 1.3 Success Metrics

- Successful MPS device detection
- GPU acceleration confirmed during training
- Compatible tensor operations across all device types
- Validated with test files 

---

## 2. Technical Challenge

### 2.1 Original Framework Limitations

**CUDA-Only Design:**
```python
# Original implementation
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
```

**Issues Identified:**
- No MPS device detection
- Float64 tensor incompatibility with MPS
- Missing device-aware tensor creation
- No command-line MPS support

### 2.2 MPS-Specific Challenges

1. **Float64 Limitation**: MPS doesn't support float64 operations
2. **Device Transfer**: Tensors need safe device transfer methods
3. **Memory Management**: Different memory tracking for MPS vs CUDA
4. **Library Compatibility**: Some packages had MPS compatibility issues

---

## 3. Implementation Strategy

### 3.1 Architecture Overview

```
Device Detection Layer
         ↓
   Optimal Device Selection (CUDA > MPS > CPU)
         ↓
   Safe Tensor Operations
         ↓
   Framework Integration
```

### 3.2 Implementation Phases

**Phase 1**: Core Device Detection

**Phase 2**: Safe Tensor Operations

**Phase 3**: Framework Integration

**Phase 4**: CLI and API Updates

**Phase 5**: Testing and Validation

---

## 4. Code Changes and Modifications

### 4.1 Core Device Detection (`utils.py`)

#### Enhanced Device Detection Function

```python
def get_optimal_device(device_preference=None):
    """
    Automatically detect and return the optimal device for PyTorch operations.
    
    Args:
        device_preference (str, optional): Preferred device type ('cuda', 'mps', 'cpu', 'auto').
                                         If None or 'auto', automatically selects the best available device.
    
    Returns:
        tuple: (device_str, device_type) where:
            - device_str: String suitable for torch.device() and PyTorch Lightning accelerator
            - device_type: String indicating the device type for compatibility
    """
    if device_preference is None:
        device_preference = 'auto'
    
    # If specific device is requested, validate and return it
    if device_preference == 'cuda':
        if torch.cuda.is_available():
            return 'cuda', 'gpu'
        else:
            warnings.warn("CUDA requested but not available. Falling back to auto-detection.")
    elif device_preference == 'mps':
        if torch.backends.mps.is_available():
            return 'mps', 'mps'
        else:
            warnings.warn("MPS requested but not available. Falling back to auto-detection.")
    elif device_preference == 'cpu':
        return 'cpu', 'cpu'
    
    # Auto-detection logic (priority: CUDA > MPS > CPU)
    if torch.cuda.is_available():
        return 'cuda', 'gpu'
    elif torch.backends.mps.is_available():
        return 'mps', 'mps'
    else:
        return 'cpu', 'cpu'
```

#### Safe Tensor Device Transfer

```python
def to_device_safe(tensor, device):
    """
    Safely move tensor to device with MPS compatibility.
    Converts float64 to float32 for MPS devices since MPS doesn't support float64.
    """
    # Handle both torch.device objects and string device names
    if isinstance(device, str):
        device = torch.device(device)
    
    if device.type == "mps" and tensor.dtype == torch.float64:
        tensor = tensor.float()  # Convert to float32
    elif device.type == "mps" and tensor.dtype == torch.double:
        tensor = tensor.float()  # Convert double to float32
    return tensor.to(device)
```

### 4.2 Command Line Interface Updates (`__main__.py`)

#### Enhanced CLI Device Support

```python
# Original
parser.add_argument('--use_gpu', action='store_true', help='Use GPU if available')

# Enhanced
parser.add_argument('--device', type=str, default='auto', 
                   choices=['auto', 'cuda', 'mps', 'cpu'],
                   help='Device to use: auto (detect optimal), cuda, mps, or cpu')

# Support legacy --use_gpu flag for backward compatibility
    if args.use_gpu and args.device == "auto":
        warnings.warn("--use_gpu is deprecated. Use --device cuda instead.", DeprecationWarning)
        device_preference = "cuda"
    else:
        device_preference = args.device
```

### 4.3 Hyperparameter Tuning Integration (`main.py`)

#### Automatic Device Detection in Training

```python
def __init__(self, dataset, model_class, config_name, target_variables, 
             device_type=None, **kwargs):
    
    self.device_type = device_type
    if self.device_type is None:
        # Use enhanced device detection
        from .utils import get_optimal_device
        _, self.device_type = get_optimal_device()
```

---

## 5. Testing and Validation

### 5.1 Test Framework Development

#### Comprehensive Test Suite

```python
import torch
from flexynesis.utils import get_optimal_device, to_device_safe

def test_flexynesis_mps_device_operations():
    """Test MPS device operations via Flexynesis enhanced device detection and safe tensor transfer."""
    device_str, device_type = get_optimal_device('mps')
    if device_str != 'mps' or not torch.backends.mps.is_available():
        print("MPS device not available via Flexynesis. Skipping test.")
        return
    device = torch.device(device_str)

    # Test basic tensor operations using safe transfer
    x = to_device_safe(torch.randn(100, 50), device)
    y = to_device_safe(torch.randn(50, 25), device)
    result = torch.mm(x, y)

    assert result.shape == (100, 25), f"Unexpected result shape: {result.shape}"
    assert result.device.type == "mps", f"Result not on MPS device: {result.device.type}"

    # Test memory tracking
    memory_before = torch.mps.current_allocated_memory()
    large_tensor = to_device_safe(torch.randn(1000, 1000), device)
    memory_after = torch.mps.current_allocated_memory()

    assert memory_after > memory_before, "Memory did not increase after allocation."
    print("Flexynesis MPS device operations test passed.")

if __name__ == "__main__":
    test_flexynesis_mps_device_operations()
```


## 6. Technical Specifications

### 6.1 System Requirements

**Flexynesis-mps seamlessly supports CUDA, MPS, and CPU devices.**

**Minimum Requirements**:
- Python 3.9+
- PyTorch 2.0+
- 8GB unified memory

**For CUDA acceleration:**
- NVIDIA GPU with CUDA support
- CUDA drivers and toolkit installed
- Linux or Windows Subsystem Linux (WSL) recommended

**For MPS acceleration:**
- macOS 12.3+ (for MPS support)
- Apple Silicon chip

**For CPU:**
- Any modern x86_64 or ARM CPU


## 7. Conclusion

Flexynesis now delivers seamless, high-performance multi-omics deep learning on CUDA, MPS, and CPU devices—empowering researchers on any platform.


