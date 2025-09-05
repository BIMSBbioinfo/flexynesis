import torch
import pytest
from flexynesis.utils import get_optimal_device, to_device_safe


def test_mps_device_detection():
    """Test MPS device detection via Flexynesis."""
    device_str, device_type = get_optimal_device('mps')
    
    if not torch.backends.mps.is_available():
        pytest.skip("MPS device not available. Skipping test.")
    
    assert device_type == 'mps', f"Expected device type 'mps', got '{device_type}'"
    assert device_str == 'mps', f"Expected device string 'mps', got '{device_str}'"


def test_mps_tensor_operations():
    """Test MPS tensor operations via Flexynesis safe transfer functions."""
    device_str, device_type = get_optimal_device('mps')
    
    if device_str != 'mps' or not torch.backends.mps.is_available():
        pytest.skip("MPS device not available. Skipping test.")
    
    device = torch.device(device_str)

    # Test basic tensor operations using safe transfer
    x = to_device_safe(torch.randn(100, 50), device)
    y = to_device_safe(torch.randn(50, 25), device)
    result = torch.mm(x, y)

    assert result.shape == (100, 25), f"Unexpected result shape: {result.shape}"
    assert result.device.type == "mps", f"Result not on MPS device: {result.device.type}"


def test_mps_memory_allocation():
    """Test MPS memory allocation tracking."""
    device_str, device_type = get_optimal_device('mps')
    
    if device_str != 'mps' or not torch.backends.mps.is_available():
        pytest.skip("MPS device not available. Skipping test.")
    
    device = torch.device(device_str)

    # Test memory tracking
    memory_before = torch.mps.current_allocated_memory()
    large_tensor = to_device_safe(torch.randn(1000, 1000), device)
    memory_after = torch.mps.current_allocated_memory()

    assert memory_after > memory_before, "Memory did not increase after allocation."


def test_float64_to_float32_conversion():
    """Test automatic float64 to float32 conversion for MPS compatibility."""
    device_str, device_type = get_optimal_device('mps')
    
    if device_str != 'mps' or not torch.backends.mps.is_available():
        pytest.skip("MPS device not available. Skipping test.")
    
    device = torch.device(device_str)

    # Test float64 tensor conversion
    x_float64 = torch.randn(10, 10, dtype=torch.float64)
    x_mps = to_device_safe(x_float64, device)

    assert x_mps.dtype == torch.float32, f"Expected float32, got {x_mps.dtype}"
    assert x_mps.device.type == "mps", f"Tensor not on MPS device: {x_mps.device.type}"
