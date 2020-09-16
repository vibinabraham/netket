# Copyright 2019 The Simons Foundation, Inc. - All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import jax.numpy as jnp
from jax import random

from .abstract_density_matrix import AbstractDensityMatrix
from ..jax import Jax as JaxPure, logcosh
from functools import partial


class Jax(JaxPure, AbstractDensityMatrix):
    def __init__(self, hilbert, module, dtype=complex, outdtype=complex):
        """
        Wraps a stax network (which is a tuple of `init_fn` and `predict_fn`)
        so that it can be used as a NetKet density matrix.

        Args:
            hilbert: Hilbert space on which the state is defined. Should be a
                subclass of `netket.hilbert.Hilbert`.
            module: A pair `(init_fn, predict_fn)`. See the documentation of
                `jax.experimental.stax` for more info.
            dtype: either complex or float, is the type used for the weights.
                In both cases the module must have a single output.
        """
        AbstractDensityMatrix.__init__(self, hilbert, dtype, outdtype)
        JaxPure.__init__(self, hilbert, module, dtype, outdtype)

        assert self.input_size == self.hilbert.size * 2

    @staticmethod
    @jax.jit
    def _dminput(xr, xc):
        if xc is None:
            x = xr
        else:
            x = jnp.hstack((xr, xc))
        return x

    def log_val(self, xr, xc=None, out=None):
        x = self._dminput(xr, xc)

        return JaxPure.log_val(self, x, out=out)

    def der_log(self, xr, xc=None, out=None):
        x = self._dminput(xr, xc)

        return JaxPure.der_log(self, x, out=out)

    def diagonal(self):
        from .diagonal import Diagonal

        diag = Diagonal(self)

        def diag_jax_forward(params, x):
            return self.jax_forward(params, self._dminput(x, x))

        diag.jax_forward = diag_jax_forward

        return diag


from jax.experimental import stax
from jax.experimental.stax import Dense
from jax.nn.initializers import glorot_normal, normal


def DensePurificationComplex(
    out_pure, out_mix, use_hidden_bias=True, W_init=glorot_normal(), b_init=normal()
):
    """Layer constructor function for a complex purification layer."""

    def init_fun(rng, input_shape):
        assert input_shape[-1] % 2 == 0
        output_shape = input_shape[:-1] + (2 * out_pure + out_mix,)

        k = jax.random.split(rng, 7)

        input_size = input_shape[-1] // 2

        # Weights for the pure part
        Wr, Wi = (
            W_init(k[0], (input_size, out_pure)),
            W_init(k[1], (input_size, out_pure)),
        )

        # Weights for the mixing part
        Vr, Vi = (
            W_init(k[2], (input_size, out_mix)),
            W_init(k[3], (input_size, out_mix)),
        )

        if use_hidden_bias:
            br, bi = (b_init(k[4], (out_pure,)), b_init(k[5], (out_pure,)))
            cr = b_init(k[6], (out_mix,))

            return output_shape, (Wr, Wi, Vr, Vi, br, bi, cr)
        else:
            return output_shape, (Wr, Wi, Vr, Vi)

    @jax.jit
    def apply_fun(params, inputs, **kwargs):
        if use_hidden_bias:
            Wr, Wi, Vr, Vi, br, bi, cr = params
        else:
            Wr, Wi, Vr, Vi = params

        xr, xc = jax.numpy.split(inputs, 2, axis=-1)

        thetar = jax.numpy.dot(
            xr[
                :,
            ],
            (Wr + 1.0j * Wi),
        )
        thetac = jax.numpy.dot(
            xc[
                :,
            ],
            (Wr - 1.0j * Wi),
        )

        thetam = jax.numpy.dot(
            xr[
                :,
            ],
            (Vr + 1.0j * Vi),
        )
        thetam += jax.numpy.dot(
            xc[
                :,
            ],
            (Vr - 1.0j * Vi),
        )

        if use_hidden_bias:
            thetar += br + 1.0j * bi
            thetac += br - 1.0j * bi
            thetam += 2 * cr

        return jax.numpy.hstack((thetar, thetam, thetac))

    return init_fun, apply_fun


from ..jax import LogCoshLayer, SumLayer


def NdmSpin(hilbert, alpha, beta, use_hidden_bias=True):
    r"""
    A fully connected Neural Density Matrix (DBM). This type density matrix is
    obtained purifying a RBM with spin 1/2 hidden units.

    The number of purification hidden units can be chosen arbitrarily.

    The weights are taken to be complex-valued. A complete definition of this
    machine can be found in Eq. 2 of Hartmann, M. J. & Carleo, G.,
    Phys. Rev. Lett. 122, 250502 (2019).

    Args:
        hilbert: Hilbert space of the system.
        alpha: `alpha * hilbert.size` is the number of hidden spins used for
                the pure state part of the density-matrix.
        beta: `beta * hilbert.size` is the number of hidden spins used for the purification.
            beta=0 for example corresponds to a pure state.
        use_hidden_bias: If ``True`` bias on the hidden units is taken.
                         Default ``True``.
    """
    return Jax(
        hilbert,
        stax.serial(
            DensePurificationComplex(
                alpha * hilbert.size, beta * hilbert.size, use_hidden_bias
            ),
            LogCoshLayer,
            SumLayer,
        ),
        dtype=float,
        outdtype=complex,
    )


def DenseMixingReal(
    out_mix, use_hidden_bias=True, W_init=glorot_normal(), b_init=normal()
):
    """Layer constructor function for a complex purification layer."""

    def init_fun(rng, input_shape):
        # Check if we are applying it to a vectorized row/col, or if they
        # are already split
        vectorised_input = True
        if len(input_shape) == 2:
            if isinstance(input_shape[0], tuple) and isinstance(input_shape[1], tuple):
                vectorised_input = False

        if vectorised_input:
            assert input_shape[-1] % 2 == 0
            output_shape = input_shape[:-1] + (out_mix,)
            input_size = input_shape[-1] // 2
        else:
            input_shape_r = input_shape[0]
            input_shape_c = input_shape[1]
            assert input_shape_r == input_shape_c
            output_shape = input_shape_r[:-1] + (out_mix,)
            input_size = input_shape_r[-1]

        k = jax.random.split(rng, 3)

        # Weights for the mixing part
        Ur, Ui = (
            W_init(k[0], (input_size, out_mix)),
            W_init(k[1], (input_size, out_mix)),
        )

        if use_hidden_bias:
            dr = b_init(k[2], (out_mix,))

            return output_shape, (Ur, Ui, dr)
        else:
            return output_shape, (Ur, Ui)

    @jax.jit
    def apply_fun(params, inputs, **kwargs):
        if use_hidden_bias:
            Ur, Ui, dr = params
        else:
            Ur, Ui = params

        # if they are already separed
        if isinstance(inputs, tuple):
            xr, xc = inputs
        else:
            xr, xc = jax.numpy.split(inputs, 2, axis=-1)

        theta = jax.numpy.dot(xr[:,], (0.5 * Ur + 0.5j * Ui))
        theta += jax.numpy.dot(xc[:,], (0.5 * Ur + 0.5j * Ui)).conj()

        if use_hidden_bias:
            theta += dr

        return theta

    return init_fun, apply_fun


def DenseMixingComplex(
    out_mix, use_hidden_bias=True, W_init=glorot_normal(), b_init=normal()
):
    """Layer constructor function for a complex purification layer."""

    def init_fun(rng, input_shape):
        assert input_shape[-1] % 2 == 0
        output_shape = input_shape[:-1] + (out_mix,)

        k = jax.random.split(rng, 3)

        input_size = input_shape[-1] // 2

        # Weights for the mixing part
        U = W_init(k[0], (input_size, out_mix))

        if use_hidden_bias:
            d = b_init(k[1], (out_mix,))

            return output_shape, (U, d)
        else:
            return output_shape, (U,)

    @jax.jit
    def apply_fun(params, inputs, **kwargs):
        if use_hidden_bias:
            U, d = params
        else:
            (U,) = params

        xr, xc = jax.numpy.split(inputs, 2, axis=-1)

        theta = jax.numpy.dot(xr[:,], U)
        theta += jax.numpy.dot(xc[:,], U).conj()

        if use_hidden_bias:
            theta += dr

        return theta

    return init_fun, apply_fun


def DensePureRowCol(
    out_pure, use_hidden_bias=True, W_init=glorot_normal(), b_init=normal()
):
    def init_fun(rng, input_shape):
        assert input_shape[-1] % 2 == 0
        input_size = input_shape[-1] // 2

        single_output_shape = input_shape[:-1] + (out_pure,)
        output_shape = (single_output_shape, single_output_shape)

        k = jax.random.split(rng, 3)

        W = W_init(k[0], (input_size, out_pure))

        if use_hidden_bias:
            b = b_init(k[2], (out_pure,))

            return output_shape, (W, b)
        else:
            return output_shape, (W,)

    def apply_fun(params, inputs, **kwargs):
        if use_hidden_bias:
            W, b = params
        else:
            W = params

        xr, xc = jax.numpy.split(inputs, 2, axis=-1)

        thetar = jax.numpy.dot(
            xr[
                :,
            ],
            W,
        )
        thetac = jax.numpy.dot(
            xc[
                :,
            ],
            W,
        )

        if use_hidden_bias:
            thetar += b
            thetac += b

        return (thetar, thetac)

    return init_fun, apply_fun


def FanInSum2():
    """Layer construction function for a fan-in sum layer."""

    def init_fun(rng, input_shape):
        output_shape = input_shape[0]
        return output_shape, tuple()

    def apply_fun(params, inputs, **kwargs):
        output = 0.5 * (inputs[0] + inputs[1])
        return output

    return init_fun, apply_fun


FanInSum2 = FanInSum2()


def FanInSub2():
    """Layer construction function for a fan-in sum layer."""

    def init_fun(rng, input_shape):
        output_shape = input_shape[0]
        return output_shape, tuple()

    def apply_fun(params, inputs, **kwargs):
        output = 0.5j * (inputs[0] - inputs[1])
        return output

    return init_fun, apply_fun


FanInSub2 = FanInSub2()


def BiasRealModPhase(b_init=normal()):
    def init_fun(rng, input_shape):
        assert input_shape[-1] % 2 == 0
        input_size = input_shape[-1] // 2

        output_shape = input_shape[:-1]

        k = jax.random.split(rng, 2)

        br = b_init(k[0], (input_size,))
        bj = b_init(k[1], (input_size,))

        return output_shape, (br, bj)

    def apply_fun(params, inputs, **kwargs):
        br, bj = params

        xr, xc = jax.numpy.split(inputs, 2, axis=-1)

        biasr = jax.numpy.dot(
            (xr + xc)[
                :,
            ],
            br,
        )
        biasj = jax.numpy.dot(
            (xr - xc)[
                :,
            ],
            bj,
        )

        return 0.5 * biasr + 0.5j * biasj

    return init_fun, apply_fun


def NdmSpinPhase(hilbert, alpha, beta, use_hidden_bias=True, use_visible_bias=True):
    r"""
    A fully connected Neural Density Matrix (DBM). This type density matrix is
    obtained purifying a RBM with spin 1/2 hidden units.

    The number of purification hidden units can be chosen arbitrarily.

    The weights are taken to be complex-valued. A complete definition of this
    machine can be found in Eq. 2 of Hartmann, M. J. & Carleo, G.,
    Phys. Rev. Lett. 122, 250502 (2019).

    Args:
        hilbert: Hilbert space of the system.
        alpha: `alpha * hilbert.size` is the number of hidden spins used for
                the pure state part of the density-matrix.
        beta: `beta * hilbert.size` is the number of hidden spins used for the purification.
            beta=0 for example corresponds to a pure state.
        use_hidden_bias: If ``True`` bias on the hidden units is taken.
                         Default ``True``.
    """
    mod_pure = stax.serial(
        DensePureRowCol(alpha * hilbert.size, use_hidden_bias),
        stax.parallel(LogCoshLayer, LogCoshLayer),
        stax.parallel(SumLayer, SumLayer),
        FanInSum2,
    )

    phs_pure = stax.serial(
        DensePureRowCol(alpha * hilbert.size, use_hidden_bias),
        stax.parallel(LogCoshLayer, LogCoshLayer),
        stax.parallel(SumLayer, SumLayer),
        FanInSub2,
    )

    mixing = stax.serial(
        DenseMixingReal(beta * hilbert.size, use_hidden_bias),
        LogCoshLayer,
        SumLayer,
    )

    if use_visible_bias:
        biases = BiasRealModPhase()
        net = stax.serial(
            stax.FanOut(4),
            stax.parallel(mod_pure, phs_pure, mixing, biases),
            stax.FanInSum,
        )
    else:
        net = stax.serial(
            stax.FanOut(3),
            stax.parallel(mod_pure, phs_pure, mixing),
            stax.FanInSum,
        )

    return Jax(hilbert, net, dtype=float, outdtype=complex)


def JaxRbmSpin(hilbert, alpha, dtype=complex):
    return Jax(
        hilbert,
        stax.serial(stax.Dense(alpha * hilbert.size * 2), LogCoshLayer, SumLayer),
        dtype=dtype,
        outdtype=dtype,
    )


# Takes a vector input and splits in two half parts run in parallel
def RowColFanOut():
    """Layer construction function for a fan-out layer splitting
    row and column indices into two parallel layers.
    Should be followed by a parallel layer."""

    def init_fun(rng, input_shape):
        assert input_shape[-1] % 2 == 0
        output_shape = input_shape[:-1] + (input_shape[-1] // 2,)

        return (output_shape, output_shape), tuple()

    def apply_fun(params, inputs, **kwargs):
        xr, xc = jax.numpy.split(inputs, 2, axis=-1)
        return xr, xc

    return init_fun, apply_fun


RowColFanOut = RowColFanOut()


def RowColSerial(*layers):
    """Combines a FanOut layer that separes row and columns into two
  different arrays, and then applies the same serial sequence of layers
  (with the exact same weights) to both, in parallel.

  Output is the output of the serial layer, for rows and columns. 
  """
    nlayers = len(layers)
    init_funs, apply_funs = zip(*layers)

    def init_fun(rng, input_shape):
        assert input_shape[-1] % 2 == 0
        rc_input_shape = input_shape[:-1] + (input_shape[-1] // 2,)

        params = []
        for init_fun in init_funs:
            rng, layer_rng = random.split(rng)
            rc_input_shape, param = init_fun(layer_rng, rc_input_shape)
            params.append(param)
        return (rc_input_shape, rc_input_shape), params

    def apply_fun(params, inputs, **kwargs):
        rng = kwargs.pop("rng", None)
        rng_row, rng_col = random.split(rng) if rng is not None else (None, None)
        rngs_row = (
            random.split(rng_row, nlayers) if rng_row is not None else (None,) * nlayers
        )
        rngs_col = (
            random.split(rng_col, nlayers) if rng_col is not None else (None,) * nlayers
        )

        inputs_row, inputs_col = jax.numpy.split(inputs, 2, axis=-1)

        for fun, param, rng in zip(apply_funs, params, rngs_row):
            inputs_row = fun(param, inputs_row, rng=rng, **kwargs)

        for fun, param, rng in zip(apply_funs, params, rngs_col):
            inputs_col = fun(param, inputs_col, rng=rng, **kwargs)

        return inputs_row, inputs_col

    return init_fun, apply_fun


def JaxDeepNetwork(
    hilbert, beta, *deep_fun, use_mixing_bias=True, activation=logcosh, dtype=float
):
    if dtype == complex:
        if use_mixing_bias:
            error("Cannot use complex parameters and mixing bias, as it is real")
        else:
            mixing = DenseMixingComplex(hilbert.size * beta, use_hidden_bias=False)
    elif dtype == float:
        mixing = DenseMixingReal(hilbert.size * beta, use_hidden_bias=use_mixing_bias)

    nnfun = stax.elementwise(activation)

    return Jax(
        hilbert,
        stax.serial(RowColSerial(*deep_fun), mixing, nnfun, SumLayer),
        dtype=dtype,
        outdtype=complex,
    )
