# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import gc
import importlib
import warnings

import torch

ALL_COMPUTE_CAPABILITIES = [20, 21, 30, 35, 37, 50, 52, 53, 60, 61, 62, 70, 72, 75, 80, 86, 89, 90]

if not torch.cuda.is_available():
    raise EnvironmentError("Unknown compute capability. Ensure PyTorch with CUDA support is installed.")

def _get_device_compute_capability(idx):
    major, minor = torch.cuda.get_device_capability(idx)
    return major * 10 + minor

def _get_system_compute_capability():
    num_devices = torch.cuda.device_count()
    device_capability = [_get_device_compute_capability(i) for i in range(num_devices)]
    system_capability = min(device_capability)

    if not all(cc == system_capability for cc in device_capability):
        warnings.warn(
            f"System has multiple GPUs with different compute capabilities: {device_capability}. "
            f"Using compute capability {system_capability} for best compatibility. "
            f"This may result in suboptimal performance."
        )
    return system_capability

# Determine the capability of the system as the minimum of all
# devices, ensuring that we have no runtime errors.
system_compute_capability = _get_system_compute_capability()

# Try to import the highest compute capability version of tcnn that
# we can find and is compatible with the system's compute capability.
_C = None
for cc in reversed(ALL_COMPUTE_CAPABILITIES):
    if cc > system_compute_capability:
        # incompatible
        continue

    try:
        _C = importlib.import_module(f"tinycudann_bindings._{cc}_C")
        if cc != system_compute_capability:
            warnings.warn(f"tinycudann was built for lower compute capability ({cc}) than the system's ({system_compute_capability}). Performance may be suboptimal.")
        break
    except ModuleNotFoundError:
        pass

if _C is None:
    raise EnvironmentError(f"Could not find compatible tinycudann extension for compute capability {system_compute_capability}.")

def _torch_precision(tcnn_precision):
    if tcnn_precision == _C.Precision.Fp16:
        return torch.half
    elif tcnn_precision == _C.Precision.Fp32:
        return torch.float
    else:
        raise ValueError(f"Unknown precision {tcnn_precision}")

def free_temporary_memory():
    # Ensure all Python objects (potentially pointing
    # to temporary TCNN allocations) are cleaned up.
    gc.collect()
    _C.free_temporary_memory()

def null_tensor_like(tensor):
    return torch.empty([], dtype=tensor.dtype, device=tensor.device)

def null_tensor_to_none(tensor):
    if len(tensor.shape) == 0:
        return None
    return tensor

def _pad(x, cast=False):
    batch_size = x.shape[0]
    batch_size_granularity = int(_C.batch_size_granularity())
    padded_batch_size = (batch_size + batch_size_granularity-1) // batch_size_granularity * batch_size_granularity
    do_cast = lambda x: x.to(torch.float32) if cast else x
    if batch_size != padded_batch_size:
        print("input requires grad", x.requires_grad)
        x = torch.nn.functional.pad(x, [0, 0, 0, padded_batch_size - batch_size])
        print("pad requires grad", x.requires_grad)
        x = do_cast(x)
        print("pad requires grad", x.requires_grad)
        x = x.contiguous()
        print("pad requires grad", x.requires_grad)
        return x
    else:
        x = do_cast(x)
        print("cast requires grad", x.requires_grad)
        return x

def _unpad(x, orig_size):
    if x is not None and x.ndim == 2:
        return x[:orig_size, :]
    else:
        return x

def _maybe_expand_bdim_at_front(x, x_bdim, batch_size):
    if x_bdim is None:
        return x.expand(batch_size, *x.shape)
    return x.movedim(x_bdim, 0)

class _module_function(torch.autograd.Function):
    @staticmethod
    def forward(native_tcnn_module, input, params, loss_scale):
        native_ctx, output = native_tcnn_module.fwd(input, params)
        return output, native_ctx

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        # If no output gradient is provided, no need to
        # automatically materialize it as torch.zeros.
        ctx.set_materialize_grads(False)

        # unpack & store forward pass inputs / outputs
        native_tcnn_module, input, params, loss_scale = inputs
        output, native_ctx = outputs
        ctx.save_for_backward(input, params, output)
        ctx.native_tcnn_module = native_tcnn_module
        ctx.native_ctx = native_ctx
        ctx.loss_scale = loss_scale

    @staticmethod
    def backward(ctx, doutput, _0):
        if doutput is None:
            return None, None, None, None

        if not doutput.is_cuda:
            warnings.warn("doutput must be a CUDA tensor, but isn't. This indicates suboptimal performance.")
            doutput = doutput.cuda()

        input, params, output = ctx.saved_tensors
        input_grad, params_grad = _module_function_backward.apply(ctx, doutput, input, params, output)

        return None, null_tensor_to_none(input_grad), null_tensor_to_none(params_grad), None

    @staticmethod
    def vmap(info, in_dims, *args):
        _, input_bddim, _, _ = in_dims
        native_tcnn_module, input, params, loss_scale = args

        # move batch dim to index 0 and flatten along that dimension
        input = _maybe_expand_bdim_at_front(input, input_bddim, info.batch_size)
        Bi, Ni, Di = input.shape
        input = input.view(Bi * Ni, Di)

        output, native_ctx = _module_function.apply(
            native_tcnn_module,
            input,
            params,
            loss_scale
        )
        # restore the batch dimension at index 0
        No, Do = output.shape
        output = output.view(Bi, Ni, Do)
        return (output, native_ctx), (0, None)

class _module_function_backward(torch.autograd.Function):
    @staticmethod
    def forward(ctx_fwd, doutput, input, params, output):
        with torch.no_grad():
            scaled_grad = doutput * ctx_fwd.loss_scale
            # print("_module_function_backward forward", input.shape, params.shape, output.shape, scaled_grad.shape)
            # print("_module_function_backward forward", input.dtype, params.dtype, output.dtype, scaled_grad.dtype)
            # print("_module_function_backward forward", input.requires_grad, params.requires_grad, output.requires_grad, scaled_grad.requires_grad)
            input_grad, params_grad = ctx_fwd.native_tcnn_module.bwd(ctx_fwd.native_ctx, input, params, output, scaled_grad)
            input_grad = null_tensor_like(input) if input_grad is None else (input_grad / ctx_fwd.loss_scale)
            params_grad = null_tensor_like(params) if params_grad is None else (params_grad / ctx_fwd.loss_scale)
            # print("_module_function_backward forward", input_grad.shape, params_grad.shape)
        return input_grad, params_grad

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        # unpack inputs & outputs
        ctx_fwd, doutput, input, params, output = inputs
        input_grad, params_grad = outputs

        ctx.ctx_fwd = ctx_fwd
        ctx.save_for_backward(input, params, doutput)

    @staticmethod
    def backward(ctx, dinput_grad, dparams_grad):
        # NOTE: currently support:
        #       ✓   d(dL_dinput)_d(dL_doutput)  doutput_grad
        #       ✓   d(dL_dinput)_d(params)      params_grad
        #       ✓   d(dL_dinput)_d(input)       input_grad
        #       x   d(dL_dparam)_d(...)
        input, params, doutput = ctx.saved_tensors
        # assert dparams_grad is None, "currently do not support 2nd-order gradients from gradient of grid"
        with torch.enable_grad():
            # NOTE: preserves requires_grad info (this function is in no_grad() context by default when invoking loss.backward())
            doutput = doutput * ctx.ctx_fwd.loss_scale
        with torch.no_grad():
            print("bwd_bwd_input")
            doutput_grad, params_grad, input_grad = ctx.ctx_fwd.native_tcnn_module.bwd_bwd_input(
                ctx.ctx_fwd.native_ctx,
                input,
                params,
                dinput_grad,
                doutput
            )
            # NOTE: be cautious when multiplying and dividing loss_scale
            #       doutput_grad uses dinput_grad
            #       params_grad  uses dinput_grad * doutput
            #       input_grad   uses dinput_grad * doutput
            params_grad = None if params_grad is None else (params_grad / ctx.ctx_fwd.loss_scale)
            input_grad = None if input_grad is None else (input_grad / ctx.ctx_fwd.loss_scale)

        # ctx_fwd,   doutput,      input,      params,      output
        return None, doutput_grad, input_grad, params_grad, None

    @staticmethod
    def vmap(info, in_dims, *fwd_args):
        _, doutput_bdim, input_bdim, _, output_bdim = in_dims
        ctx_fwd, doutput, input, params, output = fwd_args
        # print("_module_function_backward vmap", info, in_dims, doutput.shape, input.shape, params.shape, output.shape)

        # move batch dim to 0 and flatten along that dimension
        doutput = _maybe_expand_bdim_at_front(doutput, doutput_bdim, info.batch_size)
        B, N, Ddo = doutput.shape
        doutput = doutput.view(B * N, Ddo)

        input = _maybe_expand_bdim_at_front(input, input_bdim, info.batch_size)
        Di = input.shape[-1]
        input = input.view(B * N, Di)

        output = _maybe_expand_bdim_at_front(output, output_bdim, info.batch_size)
        Do = output.shape[-1]
        output = output.reshape(B * N, Do)

        # print("_module_function_backward vmap padding", doutput.shape, input.shape, output.shape)
        # print("_module_function_backward vmap padding", doutput.dtype, input.dtype, output.dtype)
        # doutput, input, output = maybe_pad(doutput, False), maybe_pad(input, True), maybe_pad(output, False)
        # print("_module_function_backward vmap padding", doutput.shape, input.shape, output.shape)
        # print("_module_function_backward vmap padding", doutput.dtype, input.dtype, output.dtype)
        input_grad, params_grad = _module_function_backward.apply(ctx_fwd, doutput, input, params, output)
        # print("_module_function_backward vmap output", input_grad.shape, params_grad.shape)

        def unpack(x):
            if x.numel() > 1:
                D = x.shape[-1]
                # crop x to remove padding and unflatten
                return (x[:B * N, ...].view(B, N, D), 0)
            return (x, None)

        input_grad, input_grad_bdim = unpack(input_grad)
        params_grad,  params_grad_bdim = params_grad.expand(info.batch_size, *params_grad.shape), 0
        return (input_grad, params_grad), (input_grad_bdim, params_grad_bdim)


class _padding_function(torch.autograd.Function):
    @staticmethod
    def forward(native_tcnn_module, input, params, loss_scale):
        batch_size = input.shape[0]
        print("_padding_function forward", input.requires_grad, params.requires_grad)
        input_padded = _pad(input, cast=True)
        print("_padding_function forward post pad", input_padded.requires_grad, params.requires_grad)
        native_ctx, output = native_tcnn_module.fwd(input_padded, params)
        output = _unpad(output, batch_size)
        print("_padding_function forward", output.requires_grad)
        return output, native_ctx

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        # If no output gradient is provided, no need to
        # automatically materialize it as torch.zeros.
        ctx.set_materialize_grads(False)

        # unpack forward pass inputs / outputs
        native_tcnn_module, input, params, loss_scale = inputs
        output, native_ctx = outputs

        # save the padded input for the backward pass instead of the original one
        ctx.save_for_backward(input, params, output)
        ctx.native_tcnn_module = native_tcnn_module
        ctx.native_ctx = native_ctx
        ctx.loss_scale = loss_scale
        ctx.batch_size = input.shape[0]

    @staticmethod
    def backward(ctx, doutput, _0):
        input, params, output = ctx.saved_tensors

        if doutput is None :
            return None, None, None, None

        # print("_padding_function bwd", doutput.shape, input.shape, input_padded.shape, params.shape, output.shape, ctx.batch_size)
        # print("_padding_function bwd", doutput.requires_grad, input.requires_grad, input_padded.requires_grad, params.requires_grad, output.requires_grad)
        if not doutput.is_cuda:
            warnings.warn("doutput must be a CUDA tensor, but isn't. This indicates suboptimal performance.")
            doutput = doutput.cuda()

        input_grad, params_grad = _padding_function_backward.apply(ctx, doutput, input, params, output)
        # print("_padding_function output", input_grad.shape, params_grad.shape)
        # input_grad = unpad(null_tensor_to_none(input_grad))
        # print("_padding_function output", input_grad.shape, params_grad.shape)
        return None, null_tensor_to_none(input_grad), null_tensor_to_none(params_grad), None

    @staticmethod
    def vmap(info, in_dims, *args):
        _, input_bddim, _, _ = in_dims
        native_tcnn_module, input, params, loss_scale = args

        input = input.movedim(input_bddim, 0)
        Bi, Ni, Di = input.shape
        input = input.view(Bi * Ni, Di)
        # print("_padding_function vmap forward", info, in_dims, input.shape, params.shape)
        input = _pad(input, cast=True)

        output, native_ctx, input_padded = _padding_function.apply(
            native_tcnn_module,
            input,
            params,
            loss_scale
        )
        No, Do = output.shape
        # print("_padding_function vmap forward output", output.shape, Bi, Ni, Do)
        output = output[:Bi * Ni, :].view(Bi, Ni, Do)
        # print("_padding_function vmap output reshape", output.shape)
        input_padded = input_padded[:Bi * Ni, :].view(Bi, Ni, Di)

        return (output, native_ctx, input_padded), (0, None, None)

class _padding_function_backward(torch.autograd.Function):
    @staticmethod
    def forward(ctx_fwd, doutput, input, params, output):
        with torch.no_grad():
            scaled_grad = doutput * ctx_fwd.loss_scale
            print("_padding_function_backward forward", input.shape, params.shape, output.shape, scaled_grad.shape)
            print("_padding_function_backward forward", input.requires_grad, params.requires_grad, output.requires_grad, scaled_grad.requires_grad)
            # print("_padding_function_backward forward", input.dtype, params.dtype, output.dtype, scaled_grad.dtype)
            # print("_padding_function_backward forward", input.requires_grad, params.requires_grad, output.requires_grad, scaled_grad.requires_grad)

            batch_size = input.shape[0]
            input = _pad(input, cast=True)
            # input.requires_grad = True
            output = _pad(output)
            scaled_grad = _pad(scaled_grad)
            print("_padding_function_backward forward", input.shape, params.shape, output.shape, scaled_grad.shape)
            print("_padding_function_backward forward", input.dtype, params.dtype, output.dtype, scaled_grad.dtype)
            input_grad, params_grad = ctx_fwd.native_tcnn_module.bwd(ctx_fwd.native_ctx, input, params, output, scaled_grad)


            input_grad = null_tensor_like(input) if input_grad is None else (input_grad / ctx_fwd.loss_scale)
            params_grad = null_tensor_like(params) if params_grad is None else (params_grad / ctx_fwd.loss_scale)
            print("_padding_function_backward output", input_grad.shape, params_grad.shape)
            input_grad = _unpad(input_grad, batch_size)
            print("_padding_function_backward output", input_grad.shape, params_grad.shape)
        return input_grad, params_grad

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        # unpack inputs & outputs
        ctx_fwd, doutput, input, params, output = inputs
        input_grad, params_grad = outputs

        ctx.ctx_fwd = ctx_fwd
        ctx.save_for_backward(input, params, doutput)

    @staticmethod
    def backward(ctx, dinput_grad, dparams_grad):
        # NOTE: currently support:
        #       ✓   d(dL_dinput)_d(dL_doutput)  doutput_grad
        #       ✓   d(dL_dinput)_d(params)      params_grad
        #       ✓   d(dL_dinput)_d(input)       input_grad
        #       x   d(dL_dparam)_d(...)
        input, params, doutput = ctx.saved_tensors
        # assert dparams_grad is None, "currently do not support 2nd-order gradients from gradient of grid"
        with torch.enable_grad():
            # NOTE: preserves requires_grad info (this function is in no_grad() context by default when invoking loss.backward())
            doutput = doutput * ctx.ctx_fwd.loss_scale
        with torch.no_grad():
            print("bwd_bwd_input")
            doutput_grad, params_grad, input_grad = ctx.ctx_fwd.native_tcnn_module.bwd_bwd_input(
                ctx.ctx_fwd.native_ctx,
                input,
                params,
                dinput_grad,
                doutput
            )
            # NOTE: be cautious when multiplying and dividing loss_scale
            #       doutput_grad uses dinput_grad
            #       params_grad  uses dinput_grad * doutput
            #       input_grad   uses dinput_grad * doutput
            params_grad = None if params_grad is None else (params_grad / ctx.ctx_fwd.loss_scale)
            input_grad = None if input_grad is None else (input_grad / ctx.ctx_fwd.loss_scale)

        # ctx_fwd,   doutput,      input,      params,      output
        return None, doutput_grad, input_grad, params_grad, None

    @staticmethod
    def vmap(info, in_dims, *fwd_args):
        _, doutput_bdim, input_bdim, _, output_bdim = in_dims
        ctx_fwd, doutput, input, params, output = fwd_args
        # print("_module_function_backward vmap", info, in_dims, doutput.shape, input.shape, params.shape, output.shape)

        # move batch dim to 0 and flatten along that dimension
        doutput = _maybe_expand_bdim_at_front(doutput, doutput_bdim, info.batch_size)
        B, N, Ddo = doutput.shape
        doutput = doutput.view(B * N, Ddo)

        input = _maybe_expand_bdim_at_front(input, input_bdim, info.batch_size)
        Di = input.shape[-1]
        input = input.view(B * N, Di)

        output = _maybe_expand_bdim_at_front(output, output_bdim, info.batch_size)
        Do = output.shape[-1]
        output = output.reshape(B * N, Do)

        # print("_module_function_backward vmap padding", doutput.shape, input.shape, output.shape)
        # print("_module_function_backward vmap padding", doutput.dtype, input.dtype, output.dtype)
        # doutput, input, output = maybe_pad(doutput, False), maybe_pad(input, True), maybe_pad(output, False)
        # print("_module_function_backward vmap padding", doutput.shape, input.shape, output.shape)
        # print("_module_function_backward vmap padding", doutput.dtype, input.dtype, output.dtype)
        input_grad, params_grad = _module_function_backward.apply(ctx_fwd, doutput, input, params, output)
        # print("_module_function_backward vmap output", input_grad.shape, params_grad.shape)

        def unpack(x):
            if x.numel() > 1:
                D = x.shape[-1]
                # crop x to remove padding and unflatten
                return (x[:B * N, ...].view(B, N, D), 0)
            return (x, None)

        input_grad, input_grad_bdim = unpack(input_grad)
        params_grad,  params_grad_bdim = params_grad.expand(info.batch_size, *params_grad.shape), 0
        return (input_grad, params_grad), (input_grad_bdim, params_grad_bdim)


class Module(torch.nn.Module):
    def __init__(self, seed=1337):
        super(Module, self).__init__()

        self.native_tcnn_module = self._native_tcnn_module()
        self.dtype = _torch_precision(self.native_tcnn_module.param_precision())

        self.seed = seed
        initial_params = self.native_tcnn_module.initial_params(seed)
        self.params = torch.nn.Parameter(initial_params, requires_grad=True)
        self.register_parameter(name="params", param=self.params)

        self.loss_scale = 128.0 if self.native_tcnn_module.param_precision() == _C.Precision.Fp16 else 1.0

    def forward(self, x):
        if not x.is_cuda:
            warnings.warn("input must be a CUDA tensor, but isn't. This indicates suboptimal performance.")
            x = x.cuda()

        batch_size = x.shape[0]
        output, _ = _padding_function.apply(
            self.native_tcnn_module,
            x,
            self.params.to(_torch_precision(self.native_tcnn_module.param_precision())).contiguous(),
            self.loss_scale,
        )


        # batch_size = x.shape[0]
        # batch_size_granularity = int(_C.batch_size_granularity())
        # padded_batch_size = (batch_size + batch_size_granularity-1) // batch_size_granularity * batch_size_granularity

        # x_padded = x if batch_size == padded_batch_size else torch.nn.functional.pad(x, [0, 0, 0, padded_batch_size - batch_size])
        # output, _ = _module_function.apply(
        #     self.native_tcnn_module,
        #     x_padded.to(torch.float).contiguous(),
        #     self.params.to(_torch_precision(self.native_tcnn_module.param_precision())).contiguous(),
        #     self.loss_scale
        # )
        return output[:batch_size, :self.n_output_dims]

    def __getstate__(self):
        """Return state values to be pickled."""
        state = self.__dict__.copy()
        # Avoid pickling native objects
        del state["native_tcnn_module"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Reconstruct native entries
        self.native_tcnn_module = self._native_tcnn_module()

    def extra_repr(self):
        return f"n_input_dims={self.n_input_dims}, n_output_dims={self.n_output_dims}, seed={self.seed}, dtype={self.dtype}, hyperparams={self.native_tcnn_module.hyperparams()}"

class NetworkWithInputEncoding(Module):
    """
    Input encoding, followed by a neural network.

    This module is more efficient than invoking individual `Encoding`
    and `Network` modules in sequence.

    Takes a `torch.float` input tensor of shape `[:, n_input_dims]` and maps
    it to a tensor of shape `[:, n_output_dims]`.

    The output tensor can be either of type `torch.float` or `torch.half`,
    depending on which performs better on the system.

    Parameters
    ----------
    n_input_dims : `int`
        Determines the shape of input tensors as `[:, n_input_dims]`
    n_output_dims : `int`
        Determines the shape of output tensors as `[:, n_output_dims]`
    encoding_config: `dict`
        Configures the encoding. Possible configurations are documented at
        https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md
    network_config: `dict`
        Configures the neural network. Possible configurations are documented at
        https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md
    seed: `int`
        Seed for pseudorandom parameter initialization
    """
    def __init__(self, n_input_dims, n_output_dims, encoding_config, network_config, seed=1337):
        if not _C.has_networks():
            raise RuntimeError(f"Cannot create `NetworkWithInputEncoding` because tiny-cuda-nn was not compiled with neural network support.")

        self.n_input_dims = n_input_dims
        self.n_output_dims = n_output_dims
        self.encoding_config = encoding_config
        self.network_config = network_config

        super(NetworkWithInputEncoding, self).__init__(seed=seed)

    def _native_tcnn_module(self):
        return _C.create_network_with_input_encoding(self.n_input_dims, self.n_output_dims, self.encoding_config, self.network_config)

class Network(Module):
    """
    Neural network.

    Takes a `torch.float` input tensor of shape `[:, n_input_dims]` and maps
    it to a tensor of shape `[:, n_output_dims]`.

    The output tensor can be either of type `torch.float` or `torch.half`,
    depending on which performs better on the system.

    Parameters
    ----------
    n_input_dims : `int`
        Determines the shape of input tensors as `[:, n_input_dims]`
    n_output_dims : `int`
        Determines the shape of output tensors as `[:, n_output_dims]`
    network_config: `dict`
        Configures the neural network. Possible configurations are documented at
        https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md
    seed: `int`
        Seed for pseudorandom parameter initialization
    """
    def __init__(self, n_input_dims, n_output_dims, network_config, seed=1337):
        if not _C.has_networks():
            raise RuntimeError(f"Cannot create `Network` because tiny-cuda-nn was not compiled with neural network support.")

        self.n_input_dims = n_input_dims
        self.n_output_dims = n_output_dims
        self.network_config = network_config

        super(Network, self).__init__(seed=seed)

    def _native_tcnn_module(self):
        return _C.create_network(self.n_input_dims, self.n_output_dims, self.network_config)

class Encoding(Module):
    """
    Input encoding to a neural network.

    Takes a `torch.float` input tensor of shape `[:, n_input_dims]` and maps
    it to a `dtype` tensor of shape `[:, self.n_output_dims]`, where
    `self.n_output_dims` depends on `n_input_dims` and the configuration
    `encoding_config`.

    Parameters
    ----------
    n_input_dims : `int`
        Determines the shape of input tensors as `[:, n_input_dims]`
    encoding_config: `dict`
        Configures the encoding. Possible configurations are documented at
        https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md
    seed: `int`
        Seed for pseudorandom parameter initialization
    dtype: `torch.dtype`
        Precision of the output tensor and internal parameters. A value
        of `None` corresponds to the optimally performing precision,
        which is `torch.half` on most systems. A value of `torch.float`
        may yield higher numerical accuracy, but is generally slower.
        A value of `torch.half` may not be supported on all systems.
    """
    def __init__(self, n_input_dims, encoding_config, seed=1337, dtype=None):
        self.n_input_dims = n_input_dims
        self.encoding_config = encoding_config
        if dtype is None:
            self.precision = _C.preferred_precision()
        else:
            if dtype == torch.float32:
                self.precision = _C.Precision.Fp32
            elif dtype == torch.float16:
                self.precision = _C.Precision.Fp16
            else:
                raise ValueError(f"Encoding only supports fp32 or fp16 precision, but got {dtype}")

        super(Encoding, self).__init__(seed=seed)

        self.n_output_dims = self.native_tcnn_module.n_output_dims()

    def _native_tcnn_module(self):
        return _C.create_encoding(self.n_input_dims, self.encoding_config, self.precision)
