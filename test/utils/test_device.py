from unittest.mock import patch, MagicMock

import pytest


@pytest.fixture
def mock_cuda_available():
    """Mock torch.cuda.is_available to return True"""
    with patch('torch.cuda.is_available', return_value=True):
        yield


@pytest.fixture
def mock_cuda_unavailable():
    """Mock torch.cuda.is_available to return False"""
    with patch('torch.cuda.is_available', return_value=False):
        yield


@pytest.fixture
def mock_torch_cuda():
    """Mock torch.cuda module"""
    mock_cuda = MagicMock()
    mock_cuda.current_device.return_value = 0
    mock_cuda.memory._set_allocator_settings = MagicMock()
    with patch('torch.cuda', mock_cuda):
        yield mock_cuda


@pytest.fixture
def mock_torch_cpu():
    """Mock torch.cpu module"""
    mock_cpu = MagicMock()
    mock_cpu.current_device.return_value = 0
    with patch('torch.cpu', mock_cpu):
        yield mock_cpu


@pytest.fixture
def mock_getattr_attribute_error():
    """Mock getattr to raise AttributeError"""

    def side_effect(obj, name):
        if name == 'test_device':
            raise AttributeError(f"module 'torch' has no attribute '{name}'")
        return getattr(obj, name)

    with patch('builtins.getattr', side_effect=side_effect):
        yield


@pytest.fixture
def mock_logger():
    """Mock logger"""
    with patch('logging.getLogger') as mock_get_logger:
        mock_logger_instance = MagicMock()
        mock_get_logger.return_value = mock_logger_instance
        yield mock_logger_instance


@pytest.mark.unittest
class TestDeviceUtils:

    def test_is_cuda_available_true(self, mock_cuda_available):
        from tprofiler.utils.device import is_cuda_available
        assert is_cuda_available() is True

    def test_is_cuda_available_false(self, mock_cuda_unavailable):
        from tprofiler.utils.device import is_cuda_available
        assert is_cuda_available() is False

    def test_get_device_name_cuda(self, mock_cuda_available):
        from tprofiler.utils.device import get_device_name
        assert get_device_name() == "cuda"

    def test_get_device_name_cpu(self, mock_cuda_unavailable):
        from tprofiler.utils.device import get_device_name
        assert get_device_name() == "cpu"

    def test_get_torch_device_cuda(self, mock_cuda_available, mock_torch_cuda):
        from tprofiler.utils.device import get_torch_device
        with patch('tprofiler.utils.device.get_device_name', return_value='cuda'):
            result = get_torch_device()
            assert result == mock_torch_cuda

    def test_get_torch_device_cpu(self, mock_cuda_unavailable, mock_torch_cpu):
        from tprofiler.utils.device import get_torch_device
        with patch('tprofiler.utils.device.get_device_name', return_value='cpu'):
            result = get_torch_device()
            assert result == mock_torch_cpu

    def test_get_device_id_cuda(self, mock_cuda_available, mock_torch_cuda):
        from tprofiler.utils.device import get_device_id
        mock_torch_cuda.current_device.return_value = 1
        with patch('tprofiler.utils.device.get_torch_device', return_value=mock_torch_cuda):
            result = get_device_id()
            assert result == 1
            mock_torch_cuda.current_device.assert_called_once()

    def test_get_device_id_cpu(self, mock_cuda_unavailable, mock_torch_cpu):
        from tprofiler.utils.device import get_device_id
        mock_torch_cpu.current_device.return_value = 0
        with patch('tprofiler.utils.device.get_torch_device', return_value=mock_torch_cpu):
            result = get_device_id()
            assert result == 0
            mock_torch_cpu.current_device.assert_called_once()

    def test_get_nccl_backend_cuda_available(self, mock_cuda_available):
        from tprofiler.utils.device import get_nccl_backend
        result = get_nccl_backend()
        assert result == "nccl"

    def test_get_nccl_backend_cuda_unavailable(self, mock_cuda_unavailable):
        from tprofiler.utils.device import get_nccl_backend
        with patch('tprofiler.utils.device.get_device_name', return_value='cpu'):
            with pytest.raises(RuntimeError, match="No available nccl backend found on device type cpu."):
                get_nccl_backend()

    def test_set_expandable_segments_cuda_available_enable(self, mock_cuda_available, mock_torch_cuda):
        from tprofiler.utils.device import set_expandable_segments
        set_expandable_segments(True)
        mock_torch_cuda.memory._set_allocator_settings.assert_called_once_with("expandable_segments:True")

    def test_set_expandable_segments_cuda_available_disable(self, mock_cuda_available, mock_torch_cuda):
        from tprofiler.utils.device import set_expandable_segments
        set_expandable_segments(False)
        mock_torch_cuda.memory._set_allocator_settings.assert_called_once_with("expandable_segments:False")

    def test_set_expandable_segments_cuda_unavailable(self, mock_cuda_unavailable):
        from tprofiler.utils.device import set_expandable_segments
        # Should not raise any error and should not call any CUDA methods
        set_expandable_segments(True)
        set_expandable_segments(False)
