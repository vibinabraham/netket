from functools import singledispatch
import numpy as _np

from netket.utils import mpi_available as _mpi_available, n_nodes as _n_nodes

if _mpi_available:
    from netket.utils import MPI_comm as _MPI_comm
    from netket.utils import MPI as _MPI


@singledispatch
def sum_inplace(x):
    """
    Computes the elementwie sum of an array or a scalar across all MPI processes.
    Attempts to perform this sum inplace if possible, but for some types a copy 
    might be returned.

    Args:
        a: The input array, which will usually be overwritten in place.
    Returns:
        out: The reduced array.
    """
    raise TypeError("Unknown type to perform dispatch upon: {}".format(type(x)))


#######
# Scalar
@sum_inplace.register(complex)
@sum_inplace.register(_np.float64)
@sum_inplace.register(_np.float32)
@sum_inplace.register(_np.complex64)
@sum_inplace.register(_np.complex128)
@sum_inplace.register(float)
def sum_inplace_scalar(a):
    ar = _np.asarray(a)

    if _n_nodes > 1:
        _MPI_comm.Allreduce(_MPI.IN_PLACE, ar.reshape(-1), op=_MPI.SUM)

    return ar


##############
# Numpy Array
#
@sum_inplace.register(_np.ndarray)
def sum_inplace_MPI(a):
    """
    Computes the elementwise sum of a numpy array over all MPI processes.

    Args:
        a (numpy.ndarray): The input array, which will be overwritten in place.
    """
    if _n_nodes > 1:
        _MPI_comm.Allreduce(_MPI.IN_PLACE, a.reshape(-1), op=_MPI.SUM)

    return a


##############
# Jax
#
from netket.utils import jax_available

if jax_available:
    import numpy as _np
    import jax

    @sum_inplace.register(jax.interpreters.xla.DeviceArray)
    def sum_inplace_jax(x):
        # if not isinstance(x, jax.interpreters.xla.DeviceArray):
        #    raise TypeError("Argument to sum_inplace_jax must be a DeviceArray, got {}"
        #            .format(type(x)))

        if _n_nodes == 1:
            return x

        # This below only works on cpus...
        # we should make this work for gpus too..
        # TODO: unsafe_buffer_pointer is considered not yet definitive interface
        ptr = x.block_until_ready().device_buffer.unsafe_buffer_pointer()

        # The above is faster.
        # This below should work more often, but might copy.
        # Depending on future changes in jaxlib, we might have to switch to
        # this below.
        # see Google/jax #2123 and #1009
        # _x = jax.xla._force(x.block_until_ready())
        # ptr = _x.device_buffer.unsafe_buffer_pointer()

        # using native numpy because jax's numpy does not have ctypeslib
        data_pointer = _np.ctypeslib.ndpointer(x.dtype, shape=x.shape)

        # wrap jax data into a standard numpy array which is handled by MPI
        arr = data_pointer(ptr).contents
        _MPI_comm.Allreduce(_MPI.IN_PLACE, arr.reshape(-1), op=_MPI.SUM)

        return x

    from jax import core
    from jax import abstract_arrays
    from jax.lib import xla_client
    from jax.interpreters import xla, ad, batching

    _ops = xla_client.ops

    ## The underlying jax primitive
    sum_inplace_p = core.Primitive("sum_inplace_mpi")  # Create the primitive

    # This function applies the primitive to a AST
    def sum_inplace_jax_primitive(x):
        return sum_inplace_p.bind(x)

    def sum_inplace_jax_value_and_jvp(in_args, tan_args):
        (x,) = in_args
        (x_tan,) = tan_args
        res = sum_inplace_jax_primitive(x)
        jvp = x_tan
        return (res, jvp)

    def sum_inplace_jax_batching(args, batch_axes):
        res = sum_inplace_p.bind(*args)
        return res, batch_axes[0]

    #  this function executes the primitive, when not under any transformation
    sum_inplace_p.def_impl(sum_inplace_jax)
    # def sum_inplace_impl(x):
    #    return sum_inplace_jax(x)
    # sum_inplace_p.def_impl(sum_inplace_impl)

    # This function evaluates only the shapes during AST construction
    def sum_inplace_abstract_eval(xs):
        return abstract_arrays.ShapedArray(xs.shape, xs.dtype)

    sum_inplace_p.def_abstract_eval(sum_inplace_abstract_eval)

    # Herlper functions
    def _constant_s32_scalar(c, x):
        return _ops.Constant(c, _np.int32(x))

    def _unpack_builder(c):
        # If `c` is a ComputationBuilder object, extracts the underlying XlaBuilder.
        return getattr(c, "_builder", c)

    #  This function compiles the operation
    def sum_inplace_xla_encode(c, x):
        c = _unpack_builder(c)
        x_shape = c.GetShape(x)
        dtype = x_shape.element_type()
        dims = x_shape.dimensions()

        # compute total number of elements in array
        nitems = dims[0]
        for el in dims[1:]:
            nitems *= el

        # those kernels have been loaded through cython.
        if dtype == _np.float32:
            kernel = b"sum_inplace_mpi_f32"
        elif dtype == _np.float64:
            kernel = b"sum_inplace_mpi_f64"
        elif dtype == _np.complex64:
            kernel = b"sum_inplace_mpi_c64"
        elif dtype == _np.complex128:
            kernel = b"sum_inplace_mpi_c128"

        return _ops.CustomCall(
            c,
            kernel,
            operands=(xla_client.ops.Constant(c, _np.int32(nitems)), x),
            shape=xla_client.Shape.array_shape(dtype, dims),
        )

    # assign to the primitive the correct encoder
    xla.backend_specific_translations["cpu"][sum_inplace_p] = sum_inplace_xla_encode
    ad.primitive_jvps[sum_inplace_p] = sum_inplace_jax_value_and_jvp
    batching.primitive_batchers[sum_inplace_p] = sum_inplace_jax_batching

    @sum_inplace.register(jax.interpreters.partial_eval.JaxprTracer)
    @sum_inplace.register(jax.interpreters.ad.JVPTracer)
    def sum_inplace_jax_jittracer(x):
        if _n_nodes == 1:
            return x
        else:
            return sum_inplace_jax_primitive(x)
