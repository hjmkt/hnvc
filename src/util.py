from jax import numpy as jnp
from flax import linen as nn
import jax


def dct1d(x):
    N = x.shape[-1]
    dct_mat = jnp.array(
        [
            jnp.array(
                [
                    jnp.sqrt(2 / N)
                    * (
                        (1 / jnp.sqrt(2) if k == 0 else 1)
                        * jnp.cos((2 * i + 1) * k * jnp.pi / (2 * N))
                    )
                    for i in range(N)
                ]
            )
            for k in range(N)
        ]
    )
    x = jnp.matmul(dct_mat, x)
    return x


def idct1d(x):
    N = x.shape[-1]
    dct_mat = jnp.array(
        [
            jnp.array(
                [
                    jnp.sqrt(2 / N)
                    * (
                        (1 / jnp.sqrt(2) if k == 0 else 1)
                        * jnp.cos((2 * i + 1) * k * jnp.pi / (2 * N))
                    )
                    for i in range(N)
                ]
            )
            for k in range(N)
        ]
    )
    idct_mat = dct_mat.T
    x = jnp.matmul(idct_mat, x)
    return x


def dct(x):
    x = dct1d(x)
    x = jnp.moveaxis(x, -1, -2)
    x = dct1d(x)
    x = jnp.moveaxis(x, -1, -2)
    return x


def idct(x):
    x = idct1d(x)
    x = jnp.moveaxis(x, -1, -2)
    x = idct1d(x)
    x = jnp.moveaxis(x, -1, -2)
    return x


def dct4x4(x):
    x = jnp.reshape(x, (x.shape[0], x.shape[1] // 4, 4, x.shape[2] // 4, 4, x.shape[3]))
    x = jnp.transpose(x, (0, 5, 1, 3, 2, 4))
    shape = x.shape
    x = jnp.reshape(x, (-1, x.shape[4], x.shape[5]))
    x = jax.vmap(dct)(x)
    x = jnp.reshape(x, shape)
    x = jnp.transpose(x, (0, 2, 3, 1, 4, 5))
    x = jnp.reshape(x, (*x.shape[:3], -1))
    return x


def idct4x4(x):
    shape = x.shape
    x = jnp.reshape(x, (-1, 4, 4))
    x = jax.vmap(idct)(x)
    x = jnp.reshape(x, (*shape[:3], 3, 4, 4))
    x = jnp.transpose(x, (0, 1, 4, 2, 5, 3))
    x = jnp.reshape(
        x, (x.shape[0], x.shape[1] * x.shape[2], x.shape[3] * x.shape[4], x.shape[5])
    )
    return x


def mish(x):
    return x * jnp.tanh(nn.softplus(x))


class LTX(nn.Module):
    @nn.compact
    def __call__(self, x, features, train=True):
        x = nn.Dense(features)(x)
        x = nn.tanh(x)
        return x


def log2(x):
    return jnp.log2(x + 1e-12)


class RateEstimator(nn.Module):
    channels: int = 9
    N: int = 16

    @nn.compact
    def __call__(self, x, bins, train=True):
        @nn.jit
        def estimate(mdl, x_ref, n):
            x, ref, scale = x_ref
            m = (jnp.arange(x.shape[0]) == n).astype(jnp.float32)
            y = x * (jnp.arange(bins) < n)
            y = jnp.concatenate([y, jnp.reshape(n / bins, (1)), ref, m, scale / 16])
            y = nn.Dense(x.shape[0] * 8)(y)
            y = mish(y)
            y = nn.LayerNorm()(y)
            y = nn.Dense(x.shape[0] * 3)(y)
            y = mish(y)
            y = nn.LayerNorm()(y)
            y = nn.Dense(x.shape[0] * 6)(y)
            y = mish(y)
            y = nn.LayerNorm()(y)
            y = nn.Dense(x.shape[0] * 2)(y)
            y = mish(y)
            y = nn.LayerNorm()(y)
            y = nn.Dense(x.shape[0] * 4)(y)
            y = mish(y)
            y = nn.LayerNorm()(y)
            # y = nn.Dense(x.shape[0] * 2)(y)
            # y = mish(y)
            # y = nn.LayerNorm()(y)
            # y = nn.Dense(x.shape[0] * 4)(y)
            # y = mish(y)
            # y = nn.LayerNorm()(y)
            y = nn.Dense(self.channels)(
                y
            )  # 1st: whether zero or not, 2nd: negative or not, 3rd~ whether less than or equal to (1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15, 19, 23, 27, 31, 39, 47, 55, 63, 79, 95, 111, 127, 159, 191, 223, 255, 319, 383, 447, 511, 639, 767, 895, 1023, 1279, 1535, 1791, 2047, 2559, 3071, 3583, 4095, 5119, 6143, 7167, 8191)
            thresh = jnp.array(
                [
                    0,
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,  #
                    16,
                    17,
                    18,
                    19,
                    20,
                    21,
                    22,
                    23,
                    24,
                    25,
                    26,
                    27,
                    28,
                    29,
                    30,
                    31,  #
                    39,
                    47,
                    55,
                    63,
                    79,
                    95,
                    111,
                    127,
                    159,
                    191,
                    223,
                    255,
                    319,
                    383,
                    447,
                    511,
                    639,
                    767,
                    895,
                    1023,
                    1279,
                    1535,
                    1791,
                    2047,
                    2559,
                    3071,
                    3583,
                    4095,
                    5119,
                    6143,
                    7167,
                    8191,
                ][: self.channels]
            )
            p = nn.sigmoid(y)  # probability of each flag to be 1

            v = jnp.abs(x[n])
            m = jnp.clip(jnp.argmax(thresh > v) - 1, 2, self.channels - 1)
            # d = thresh[m] - thresh[m - 1]
            mask = (jnp.arange(self.channels) >= 2).astype(jnp.float32) * (
                jnp.arange(self.channels) < m
            ).astype(jnp.float32)

            bits = jax.lax.cond(
                x[n] == 0,
                lambda: -log2(p[0]),
                lambda: -log2(1 - p[0])
                + jax.lax.cond(
                    x[n] < 0,
                    lambda: -log2(p[1]),
                    lambda: -log2(1 - p[1]),
                )
                + (-log2(1 - p * mask)).sum()
                + -log2(p[m])
                # + jax.lax.cond(m <= 15, lambda: 0.0, lambda: log2(d)),
            )

            return (x_ref, bits)

        # N = x.shape[0] // 6
        x, ref, scale = (
            x[: self.N],
            x[self.N : -self.N],
            x[-self.N :],
        )
        f = nn.scan(estimate, variable_broadcast="params", split_rngs={"params": False})
        indices = jnp.arange(bins)
        if self.is_mutable_collection("params"):
            (_, bits) = estimate(self, (x, ref, scale), indices[0])
        else:
            (_, bits) = f(self, (x, ref, scale), indices)
            bits = bits.sum()

        return bits


class MetricsLogger:
    def __init__(self, metrics, train):
        self.epoch = 0
        self.itr = 0
        self.train = train
        self.metrics = {}
        for m in metrics:
            self.metrics[m] = 0.0

    def get(self, metric):
        return self.metrics[metric] / self.itr

    def update(self, epoch, metrics, bar=None):
        if epoch != self.epoch:
            self.epoch = epoch
            self.itr = 0
            for m in self.metrics.keys():
                self.metrics[m] = 0.0
        for k, v in metrics.items():
            self.metrics[k] += v
        self.itr += 1
        if bar is not None:
            self.update_bar_postfix(bar)
            bar.update(1)

    def update_bar_postfix(self, bar, max_len=7):
        postfix = {}
        for k, v in self.metrics.items():
            if hasattr(v, "numpy"):
                v = v.numpy()
            postfix[f"{k}"] = "{:f}".format(v / (self.itr))[:max_len]
        bar.set_postfix(**postfix)

    def set_summary(self, summary_writer):
        label = "train" if self.train else "test"
        for k, v in self.metrics.items():
            summary_writer.scalar(f"{label}/{k}", v / self.itr, self.epoch)
