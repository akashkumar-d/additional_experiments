"""
Teacher activation functions used by the M3 (1-HL ReLU NN) experiments.

Only ReLU, GELU, LeakyReLU, and SiLU are exposed here, matching the configs
in `nn_m3/`. The student is always ReLU; the activation here only changes
the teacher's labels.
"""
import math
import numpy as np


def _erf_np(z):
    """Abramowitz & Stegun 7.1.26 erf approximation, ~1.5e-7 accuracy."""
    a = [0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429]
    p = 0.3275911
    sign = np.sign(z); z = np.abs(z)
    t = 1.0 / (1.0 + p * z)
    y = 1.0 - (((((a[4]*t + a[3])*t) + a[2])*t + a[1])*t + a[0])*t * np.exp(-z*z)
    return sign * y


def relu_act(z):
    return np.maximum(z, 0.0)


def leaky_relu_act(z, alpha=0.1):
    return np.where(z > 0, z, alpha * z)


def gelu_act(z):
    """Exact GELU: 0.5 z (1 + erf(z/sqrt 2))."""
    return 0.5 * z * (1.0 + _erf_np(z / math.sqrt(2.0)))


def silu_act(z):
    """SiLU/Swish: z * sigmoid(z)."""
    return z * (1.0 / (1.0 + np.exp(-np.clip(z, -50, 50))))


ACTIVATIONS = {
    "relu":      relu_act,
    "leakyrelu": leaky_relu_act,
    "gelu":      gelu_act,
    "silu":      silu_act,
}
