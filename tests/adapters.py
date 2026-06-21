from __future__ import annotations

import torch
from torch import distributed as dist


def get_flashattention_autograd_function_pytorch() -> type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention2.
    The expectation is that this class will implement FlashAttention2
    using only standard PyTorch operations (no Triton!).

    Returns:
        A class object (not an instance of the class)
    """
    # For example: return MyFlashAttnAutogradFunctionClass
    # raise NotImplementedError
    from cs336_systems.flash_attention.flash_att_pytorch import flash_attention_pytorch
    return flash_attention_pytorch


def get_flashattention_autograd_function_triton() -> type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention2
    using Triton kernels.
    The expectation is that this class will implement the same operations
    as the class you return in get_flashattention_autograd_function_pytorch(),
    but it should do so by invoking custom Triton kernels in the forward
    and backward passes.

    Returns:
        A class object (not an instance of the class)
    """
    # For example: return MyTritonFlashAttentionAutogradFunctionClass
    # raise NotImplementedError
    from cs336_systems.flash_attention.flash_att_triton import flash_attention_triton
    return flash_attention_triton



def get_ddp(module: torch.nn.Module) -> torch.nn.Module:
    """
    Returns a torch.nn.Module container that handles
    parameter broadcasting and gradient synchronization for
    distributed data parallel training.

    This container should overlaps communication with backprop computation
    by asynchronously communicating gradients as they are ready
    in the backward pass. The gradient for each parameter tensor
    is individually communicated.

    Args:
        module: torch.nn.Module
            Underlying model to wrap with DDP.
    Returns:
        Instance of a DDP class.
    """
    # For example: return DDP(module)
    # raise NotImplementedError
    # from cs336_systems.ddp.naive_ddp import naiveddp
    # return naiveddp(module)
    from cs336_systems.ddp.ddp_overlap_individual_parameters import DDPOverlapIndividualParameters
    return DDPOverlapIndividualParameters(module)




def ddp_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    # For example: ddp_model.finish_gradient_synchronization()
    # raise NotImplementedError
    # ddp_model.finish_gradient_synchronization()
    if hasattr(ddp_model, "finish_gradient_synchronization"):
        ddp_model.finish_gradient_synchronization()


def get_fsdp(module: torch.nn.Module, compute_dtype: torch.dtype | None = None) -> torch.nn.Module:
    """
    Returns a torch.nn.Module container that handles
    fully-sharded data parallel training, including weight sharding,
    all-gather for forward/backward, and gradient reduce-scatter.

    Args:
        module: torch.nn.Module
            Underlying model to wrap with FSDP.
        compute_dtype: optional torch.dtype
            If provided, weights are cast to this dtype before communication
            and compute, saving bandwidth. Master weights stay in fp32.
    Returns:
        Instance of an FSDP class.
    """
    # For example: return FSDP(module, compute_dtype=compute_dtype)
    # raise NotImplementedError
    from cs336_systems.fsdp.fsdp import FSDP
    return FSDP(module, compute_dtype=compute_dtype)


def fsdp_on_after_backward(fsdp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.

    Args:
        fsdp_model: torch.nn.Module
            FSDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the FSDP-wrapped model.
    """
    # For example: fsdp_model.finish_gradient_synchronization()
    # raise NotImplementedError
    from cs336_systems.fsdp.fsdp import FSDP
    fsdp_model.finish_gradient_synchronization()


def fsdp_gather_full_params(fsdp_model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """
    All-gather sharded parameters from the FSDP model to reconstruct full
    parameter tensors. Replicated parameters are returned as-is.

    Args:
        fsdp_model: torch.nn.Module
            FSDP-wrapped model.
    Returns:
        State dictionary mapping parameter names to full (unsharded) tensors.
    """
    # raise NotImplementedError
    state_dict = {}
    
    for name, param in fsdp_model.module.named_parameters():
        param_id = id(param)
        if param_id in fsdp_model._param_info:
            # This is a sharded parameter - need to all-gather
            info = fsdp_model._param_info[param_id]
            sharded = info['sharded']
            
            # All-gather the shards
            shards = [torch.empty_like(sharded.data) for _ in range(fsdp_model.world_size)]
            torch.dist.all_gather(shards, sharded.data)
            full = torch.cat(shards, dim=0)
            
            state_dict[name] = full
        else:
            # This is a replicated parameter (like RMSNorm) - return as-is
            state_dict[name] = param.data.clone()
    
    return state_dict




def get_sharded_optimizer(params, optimizer_cls: type[torch.optim.Optimizer], **kwargs) -> torch.optim.Optimizer:
    """
    Returns a torch.optim.Optimizer that handles optimizer state sharding
    of the given optimizer_cls on the provided parameters.

    Arguments:
        params (``Iterable``): an ``Iterable`` of :class:`torch.Tensor` s
            or :class:`dict` s giving all parameters, which will be sharded
            across ranks.
        optimizer_class (:class:`torch.nn.Optimizer`): the class of the local
            optimizer.
    Keyword arguments:
        kwargs: keyword arguments to be forwarded to the optimizer constructor.
    Returns:
        Instance of sharded optimizer.
    """
    # raise NotImplementedError
    from cs336_systems.ddp.shared_optimizer import SharedOptimizer
    return SharedOptimizer(params, optimizer_cls, **kwargs)
