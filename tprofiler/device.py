"""
Device utilities for PyTorch operations.

This module provides utilities for device detection, management, and configuration
across different hardware backends including CPU, CUDA, and NPU. It handles device
selection, torch device namespace retrieval, device ID management, and NCCL backend
configuration for distributed training scenarios.

The module is inspired by torchtune's device utilities and provides a unified
interface for working with different device types in PyTorch applications.
"""

# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# This code is inspired by the torchtune.
# https://github.com/pytorch/torchtune/blob/main/torchtune/utils/_device.py
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license in https://github.com/pytorch/torchtune/blob/main/LICENSE

import logging

import torch

logger = logging.getLogger(__name__)


def is_cuda_available() -> bool:
    return torch.cuda.is_available()


def get_device_name() -> str:
    """
    Get the torch.device name based on the current machine's available hardware.

    This function determines the appropriate device type for PyTorch operations
    by checking hardware availability. Currently supports CPU and CUDA devices.
    NPU support has been removed from this implementation.

    :return: The device name string ('cuda' if CUDA is available, otherwise 'cpu').
    :rtype: str

    Example::
        >>> device_name = get_device_name()
        >>> print(device_name)  # 'cuda' or 'cpu'
    """
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return device


def get_torch_device() -> any:
    """
    Return the corresponding torch device namespace based on the current device type.

    This function retrieves the appropriate torch device module (e.g., torch.cuda)
    based on the detected device type. It provides a unified way to access
    device-specific PyTorch functionality.

    :return: The corresponding torch device namespace module.
    :rtype: any

    Example::
        >>> device_module = get_torch_device()
        >>> # Returns torch.cuda if CUDA is available, torch.cpu otherwise
    """
    device_name = get_device_name()
    try:
        return getattr(torch, device_name)
    except AttributeError:
        logger.warning(f"Device namespace '{device_name}' not found in torch, try to load torch.cuda.")
        return torch.cuda


def get_device_id() -> int:
    """
    Return the current device ID based on the detected device type.

    This function retrieves the current device index for the active device type.
    For CUDA devices, this returns the current CUDA device ID. For CPU, this
    typically returns 0.

    :return: The current device index.
    :rtype: int

    Example::
        >>> device_id = get_device_id()
        >>> print(f"Current device ID: {device_id}")
    """
    return get_torch_device().current_device()


def get_nccl_backend() -> str:
    """
    Return the appropriate NCCL backend type based on the current device type.

    This function determines the correct NCCL (NVIDIA Collective Communications Library)
    backend for distributed training operations. NCCL is primarily used for
    multi-GPU communication in distributed PyTorch training.

    :return: The NCCL backend type string.
    :rtype: str
    :raises RuntimeError: If no available NCCL backend is found for the current device type.

    Example::
        >>> try:
        ...     backend = get_nccl_backend()
        ...     print(f"NCCL backend: {backend}")
        ... except RuntimeError as e:
        ...     print(f"Error: {e}")
    """
    if torch.cuda.is_available():
        return "nccl"
    else:
        raise RuntimeError(f"No available nccl backend found on device type {get_device_name()}.")


def set_expandable_segments(enable: bool) -> None:
    """
    Enable or disable expandable segments for CUDA memory allocation.

    This function configures CUDA memory allocator settings to use expandable
    segments, which can help avoid out-of-memory (OOM) errors by allowing
    the memory pool to grow dynamically. This is particularly useful for
    training large models or handling variable batch sizes.

    :param enable: Whether to enable expandable segments for memory allocation.
    :type enable: bool

    Example::
        >>> # Enable expandable segments to help avoid OOM
        >>> set_expandable_segments(True)
        >>> 
        >>> # Disable expandable segments for more predictable memory usage
        >>> set_expandable_segments(False)
    """
    if torch.cuda.is_available():
        torch.cuda.memory._set_allocator_settings(f"expandable_segments:{enable}")
