"""
Distributed tensor gathering utilities for PyTorch.

This module provides utilities for gathering tensors across multiple processes in a distributed
PyTorch environment. It handles tensors of different sizes along the concatenation dimension
by using padding and unpadding strategies.
"""

from typing import Optional

import torch
import torch.distributed as dist


def gather(
        tensor: torch.Tensor,
        dim: int = 0,
        dst: Optional[int] = None
) -> Optional[torch.Tensor]:
    """
    Gather tensors from all processes in a distributed environment.

    This function collects tensors from all processes and concatenates them along the specified
    dimension. It handles tensors of different sizes by padding them to the same size before
    gathering and then removing the padding after concatenation.

    :param tensor: The tensor to gather from the current process.
    :type tensor: torch.Tensor
    :param dim: The dimension along which to concatenate the gathered tensors. Defaults to 0.
    :type dim: int
    :param dst: The destination rank that should receive the gathered result. If None, all ranks
                receive the result (all_gather mode). If specified, only the destination rank
                receives the result (gather mode).
    :type dst: Optional[int]

    :return: The concatenated tensor from all processes if this rank should receive the result,
             None otherwise. In non-distributed environments, returns the input tensor unchanged.
    :rtype: Optional[torch.Tensor]

    Example::

        >>> # All-gather mode: all ranks get the result
        >>> local_tensor = torch.tensor([1, 2, 3])
        >>> result = gather(local_tensor, dim=0)
        >>> # result will be concatenation of tensors from all ranks

        >>> # Gather mode: only rank 0 gets the result
        >>> local_tensor = torch.tensor([[1, 2], [3, 4]])
        >>> result = gather(local_tensor, dim=0, dst=0)
        >>> # Only rank 0 will have the concatenated result, others get None
    """
    if not dist.is_initialized():
        return tensor

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if world_size == 1:
        return tensor

    # 1. Collect the sizes of tensors along the concatenation dimension from all ranks
    concat_dim_size = torch.tensor([tensor.shape[dim]], dtype=torch.long, device=tensor.device)
    all_concat_dim_sizes = [torch.zeros_like(concat_dim_size) for _ in range(world_size)]
    dist.all_gather(all_concat_dim_sizes, concat_dim_size)

    # Convert to python list for easier subsequent use
    all_concat_dim_sizes = [size.item() for size in all_concat_dim_sizes]

    # 2. To use gather, need to pad all tensors to the same size
    max_concat_dim_size = max(all_concat_dim_sizes)

    # Create padded tensor
    padded_shape = list(tensor.shape)
    padded_shape[dim] = max_concat_dim_size
    padded_tensor = torch.zeros(padded_shape, dtype=tensor.dtype, device=tensor.device)

    # Copy original tensor into padded tensor
    slices = [slice(None)] * tensor.ndim
    slices[dim] = slice(0, tensor.shape[dim])
    padded_tensor[tuple(slices)] = tensor

    # 3. Execute gather operation
    if dst is None:
        # all_gather mode: all ranks get the result
        gathered_tensors = [torch.zeros_like(padded_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_tensors, padded_tensor)
        should_return_result = True
    else:
        # gather mode: only target rank gets the result
        if rank == dst:
            gathered_tensors = [torch.zeros_like(padded_tensor) for _ in range(world_size)]
        else:
            gathered_tensors = None

        dist.gather(padded_tensor, gathered_tensors, dst=dst)
        should_return_result = (rank == dst)

    # 4. If current rank should return result, remove padding and concatenate
    if should_return_result:
        # Remove padding
        unpadded_tensors = []
        for i, gathered_tensor in enumerate(gathered_tensors):
            slices = [slice(None)] * gathered_tensor.ndim
            slices[dim] = slice(0, all_concat_dim_sizes[i])
            unpadded_tensor = gathered_tensor[tuple(slices)]
            unpadded_tensors.append(unpadded_tensor)

        # Concatenate along specified dimension
        result = torch.cat(unpadded_tensors, dim=dim)
        return result
    else:
        return None
