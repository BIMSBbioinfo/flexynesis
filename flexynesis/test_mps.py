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
