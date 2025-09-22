from typing import Optional

import torch
import torch.distributed as dist


def gather(
        tensor: torch.Tensor,
        dim: int = 0,
        dst: Optional[int] = None
) -> Optional[torch.Tensor]:
    if not dist.is_initialized():
        return tensor

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if world_size == 1:
        return tensor

    # 1. 收集所有rank上tensor在concat维度的大小
    concat_dim_size = torch.tensor([tensor.shape[dim]], dtype=torch.long, device=tensor.device)
    all_concat_dim_sizes = [torch.zeros_like(concat_dim_size) for _ in range(world_size)]
    dist.all_gather(all_concat_dim_sizes, concat_dim_size)

    # 转换为python list便于后续使用
    all_concat_dim_sizes = [size.item() for size in all_concat_dim_sizes]

    # 2. 为了使用gather，需要将所有tensor padding到相同大小
    max_concat_dim_size = max(all_concat_dim_sizes)

    # 创建padding后的tensor
    padded_shape = list(tensor.shape)
    padded_shape[dim] = max_concat_dim_size
    padded_tensor = torch.zeros(padded_shape, dtype=tensor.dtype, device=tensor.device)

    # 将原tensor复制到padded_tensor中
    slices = [slice(None)] * tensor.ndim
    slices[dim] = slice(0, tensor.shape[dim])
    padded_tensor[tuple(slices)] = tensor

    # 3. 执行gather操作
    if dst is None:
        # all_gather模式：所有rank都获得结果
        gathered_tensors = [torch.zeros_like(padded_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_tensors, padded_tensor)
        should_return_result = True
    else:
        # gather模式：只有目标rank获得结果
        if rank == dst:
            gathered_tensors = [torch.zeros_like(padded_tensor) for _ in range(world_size)]
        else:
            gathered_tensors = None

        dist.gather(padded_tensor, gathered_tensors, dst=dst)
        should_return_result = (rank == dst)

    # 4. 如果当前rank应该返回结果，则去除padding并concat
    if should_return_result:
        # 去除padding
        unpadded_tensors = []
        for i, gathered_tensor in enumerate(gathered_tensors):
            slices = [slice(None)] * gathered_tensor.ndim
            slices[dim] = slice(0, all_concat_dim_sizes[i])
            unpadded_tensor = gathered_tensor[tuple(slices)]
            unpadded_tensors.append(unpadded_tensor)

        # 在指定维度上concat
        result = torch.cat(unpadded_tensors, dim=dim)
        return result
    else:
        return None
