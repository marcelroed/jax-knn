# -*- coding: utf-8 -*-

__all__ = ["kdknn"]

from functools import partial

import numpy as np
from jax import numpy as jnp
from jax.lib import xla_client
from jax import core, dtypes, lax
from jax.interpreters import ad, batching, xla
from jax.abstract_arrays import ShapedArray

# Register the CPU XLA custom calls
from . import cpu_ops

for _name, _value in cpu_ops.registrations().items():
    xla_client.register_cpu_custom_call_target(_name, _value)

# If the GPU version exists, also register those
try:
    from . import gpu_ops
except ImportError:
    gpu_ops = None
else:
    for _name, _value in gpu_ops.registrations().items():
        xla_client.register_custom_call_target(_name, _value, platform="gpu")

xops = xla_client.ops


# This function exposes the primitive to user code and this is the only
# public-facing function in this module
def kdknn(mean_anom, ecc):
    # We're going to apply array broadcasting here since the logic of our op
    # is much simpler if we require the inputs to all have the same shapes
    mean_anom_, ecc_ = jnp.broadcast_arrays(mean_anom, ecc)

    # Then we need to wrap into the range [0, 2*pi)
    M_mod = jnp.mod(mean_anom_, 2 * np.pi)

    return _kdknn_prim.bind(M_mod, ecc_)


# *********************************
# *  SUPPORT FOR JIT COMPILATION  *
# *********************************

# For JIT compilation we need a function to evaluate the shape and dtype of the
# outputs of our op for some given inputs
def _kdknn_abstract(mean_anom, ecc):
    shape = mean_anom.shape
    dtype = dtypes.canonicalize_dtype(mean_anom.dtype)
    assert dtypes.canonicalize_dtype(ecc.dtype) == dtype
    assert ecc.shape == shape
    return (ShapedArray(shape, dtype), ShapedArray(shape, dtype))


# We also need a translation rule to convert the function into an XLA op. In
# our case this is the custom XLA op that we've written. We're wrapping two
# translation rules into one here: one for the CPU and one for the GPU
def _kdknn_translation(c, mean_anom, ecc, *, platform="cpu"):
    # The inputs have "shapes" that provide both the shape and the dtype
    mean_anom_shape = c.get_shape(mean_anom)
    ecc_shape = c.get_shape(ecc)

    # Extract the dtype and shape
    dtype = mean_anom_shape.element_type()
    dims = mean_anom_shape.dimensions()
    assert ecc_shape.element_type() == dtype
    assert ecc_shape.dimensions() == dims

    # The total size of the input is the product across dimensions
    size = np.prod(dims).astype(np.int64)

    # The inputs and outputs all have the same shape so let's predefine this
    # specification
    shape = xla_client.Shape.array_shape(
        np.dtype(dtype), dims, tuple(range(len(dims) - 1, -1, -1))
    )

    # We dispatch a different call depending on the dtype
    if dtype == np.float32:
        op_name = platform.encode() + b"_kdknn_f32"
    elif dtype == np.float64:
        op_name = platform.encode() + b"_kdknn_f64"
    else:
        raise NotImplementedError(f"Unsupported dtype {dtype}")

    # And then the following is what changes between the GPU and CPU
    if platform == "cpu":
        # On the CPU, we pass the size of the data as a the first input
        # argument
        return xops.CustomCallWithLayout(
            c,
            op_name,
            # The inputs:
            operands=(xops.ConstantLiteral(c, size), mean_anom, ecc),
            # The input shapes:
            operand_shapes_with_layout=(
                xla_client.Shape.array_shape(np.dtype(np.int64), (), ()),
                shape,
                shape,
            ),
            # The output shapes:
            shape_with_layout=xla_client.Shape.tuple_shape((shape, shape)),
        )

    elif platform == "gpu":
        if gpu_ops is None:
            raise ValueError(
                "The 'kdknn_jax' module was not compiled with CUDA support"
            )

        # On the GPU, we do things a little differently and encapsulate the
        # dimension using the 'opaque' parameter
        opaque = gpu_ops.build_kdknn_descriptor(size)

        return xops.CustomCallWithLayout(
            c,
            op_name,
            operands=(mean_anom, ecc),
            operand_shapes_with_layout=(shape, shape),
            shape_with_layout=xla_client.Shape.tuple_shape((shape, shape)),
            opaque=opaque,
        )

    raise ValueError(
        "Unsupported platform; this must be either 'cpu' or 'gpu'"
    )


# **********************************
# *  SUPPORT FOR FORWARD AUTODIFF  *
# **********************************

# Here we define the differentiation rules using a JVP derived using implicit
# differentiation of kdknn's equation:
#
#  M = E - e * sin(E)
#  -> dM = dE * (1 - e * cos(E)) - de * sin(E)
#  -> dE/dM = 1 / (1 - e * cos(E))  and  de/dM = sin(E) / (1 - e * cos(E))
#
# In this case we don't need to define a transpose rule in order to support
# reverse and higher order differentiation. This might not be true in other
# applications, so check out the "How JAX primitives work" tutorial in the JAX
# documentation for more info as necessary.
def _kdknn_jvp(args, tangents):
    mean_anom, ecc = args
    d_mean_anom, d_ecc = tangents

    # We use "bind" here because we don't want to mod the mean anomaly again
    sin_ecc_anom, cos_ecc_anom = _kdknn_prim.bind(mean_anom, ecc)

    def zero_tangent(tan, val):
        return lax.zeros_like_array(val) if type(tan) is ad.Zero else tan

    # Propagate the derivatives
    d_ecc_anom = (
        zero_tangent(d_mean_anom, mean_anom)
        + zero_tangent(d_ecc, ecc) * sin_ecc_anom
    ) / (1 - ecc * cos_ecc_anom)

    return (sin_ecc_anom, cos_ecc_anom), (
        cos_ecc_anom * d_ecc_anom,
        -sin_ecc_anom * d_ecc_anom,
    )


# ************************************
# *  SUPPORT FOR BATCHING WITH VMAP  *
# ************************************

# Our op already supports arbitrary dimensions so the batching rule is quite
# simple. The jax.lax.linalg module includes some example of more complicated
# batching rules if you need such a thing.
def _kdknn_batch(args, axes):
    assert axes[0] == axes[1]
    return kdknn(*args), axes


# *********************************************
# *  BOILERPLATE TO REGISTER THE OP WITH JAX  *
# *********************************************
_kdknn_prim = core.Primitive("kdknn")
_kdknn_prim.multiple_results = True
_kdknn_prim.def_impl(partial(xla.apply_primitive, _kdknn_prim))
_kdknn_prim.def_abstract_eval(_kdknn_abstract)

# Connect the XLA translation rules for JIT compilation
xla.backend_specific_translations["cpu"][_kdknn_prim] = partial(
    _kdknn_translation, platform="cpu"
)
xla.backend_specific_translations["gpu"][_kdknn_prim] = partial(
    _kdknn_translation, platform="gpu"
)

# Connect the JVP and batching rules
ad.primitive_jvps[_kdknn_prim] = _kdknn_jvp
batching.primitive_batchers[_kdknn_prim] = _kdknn_batch
