import sys
import subprocess
import torch

def test_flexynesis_mps_device_operations_from_wheel():
    """Test MPS device operations via Flexynesis installed from wheel."""
    # Install the wheel package
    wheel_path = "/Users/hc/Documents/uber/flexynesis-mps/dist/flexynesis_mps-1.0.0-py3-none-any.whl"
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--force-reinstall", wheel_path])

    from flexynesis.utils import get_optimal_device, to_device_safe
    device_str, device_type = get_optimal_device('mps')
    if device_str != 'mps' or not torch.backends.mps.is_available():
        print("MPS device not available via Flexynesis (wheel). Skipping test.")
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
    print("Flexynesis (wheel) MPS device operations test passed.")

if __name__ == "__main__":
    test_flexynesis_mps_device_operations_from_wheel()
