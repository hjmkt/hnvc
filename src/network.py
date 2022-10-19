import jax
from flax import linen as nn
from util import RateEstimator, MetricsLogger, mish, dct, idct, LTX
from jax import numpy as jnp


def batch_norm(x, train):
    return nn.BatchNorm(use_running_average=not train, use_bias=False, use_scale=False)(
        x
    )


def layer_norm(x):
    return nn.LayerNorm()(x)


def instance_norm(x):
    return nn.GroupNorm(num_groups=None, group_size=1)(x)


def group_norm(x, group_size=4):
    return nn.GroupNorm(num_groups=None, group_size=group_size)(x)


class ResBlock(nn.Module):
    features: int
    ksize: int

    @nn.compact
    def __call__(self, x, train):
        if x.shape[-1] == self.features:
            r = x
        else:
            r = nn.Conv(self.features, (1, 1))(x)
        x = layer_norm(x)
        x = mish(x)
        x = nn.Conv(self.features, (self.ksize, self.ksize))(x)
        x = layer_norm(x)
        x = mish(x)
        x = nn.Conv(self.features, (self.ksize, self.ksize))(x)
        x = x + r
        return x


class ConvNeXtBlock(nn.Module):
    features: int
    ksize: int

    @nn.compact
    def __call__(self, x, train, norm=None, dw=True):
        if x.shape[-1] == self.features:
            r = x
        else:
            r = nn.Conv(self.features, (1, 1))(x)
        x = nn.Conv(
            self.features,
            (self.ksize, self.ksize),
            feature_group_count=x.shape[-1] if dw else 1,
        )(x)
        if norm is not None:
            x = norm(x)
        x = nn.Conv(self.features * 4, (1, 1))(x)
        x = mish(x)
        x = nn.Conv(self.features, (1, 1))(x)
        x = x + r
        return x


def conv(x, features, ksize, strides, activation, norm):
    x = nn.Conv(features, (ksize, ksize), strides)(x)
    x = activation(x)
    if norm is not None:
        x = norm(x)
    return x


def conv_transpose(x, features, ksize, strides, activation, norm):
    x = nn.ConvTranspose(features, (ksize, ksize), (strides, strides))(x)
    x = activation(x)
    if norm is not None:
        x = norm(x)
    return x


def pixel_shuffler(x):
    x = conv(x, x.shape[-1] * 4, 1, 1, mish, group_norm)
    x = jnp.reshape(x, (*x.shape[:3], 2, 2, -1))
    x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))
    x = jnp.reshape(
        x, (x.shape[0], x.shape[1] * x.shape[2], x.shape[3] * x.shape[4], -1)
    )
    return x


def conv_next_blocks(x, features, ksize, num_blocks, train, norm, dw=True):
    for _ in range(num_blocks):
        x = ConvNeXtBlock(features, ksize)(x, train, norm, dw)
    return x


def res_blocks(x, features, ksize, num_blocks, train, norm, dw=True):
    for _ in range(num_blocks):
        x = ResBlock(features, ksize)(x, train)
    return x


class Encoder(nn.Module):
    channels: int = 16
    mid_channels: int = 32
    ksize: int = 3
    mid_ksize: int = 3
    avg_scale: int = 4
    max_scale: int = 128
    blocks: int = 3
    net = "conv"

    @nn.compact
    def __call__(self, x, train, rc=False):
        if self.net == "conv":
            x = conv(x, self.mid_channels, self.mid_ksize, 2, mish, layer_norm)
            x = res_blocks(
                x, self.mid_channels, self.mid_ksize, self.blocks, train, layer_norm
            )
            x = conv(x, self.mid_channels, self.mid_ksize, 2, mish, layer_norm)
            x = res_blocks(
                x, self.mid_channels, self.mid_ksize, self.blocks, train, layer_norm
            )
        x = conv(x, self.mid_channels, self.ksize, 1, mish, layer_norm)
        e = conv(x, self.channels, self.ksize, 1, nn.sigmoid, None)

        scale = conv(e, self.channels, 3, 1, nn.sigmoid, None)
        # scale = self.min_scale * jnp.exp(
        # scale * jnp.log(self.max_scale // self.min_scale)
        # )

        if rc is None:
            scale = jnp.mean(scale, axis=(1, 2))
            scale = jnp.minimum(
                scale / jnp.mean(scale, axis=-1, keepdims=True) * self.avg_scale,
                self.max_scale,
            )
            scale = jnp.reshape(scale, (scale.shape[0], 1, 1, -1))
            scale = jnp.tile(scale, (1, e.shape[1], e.shape[2], 1))
        else:
            a = jnp.mean(scale, axis=-1, keepdims=True)
            a = a / jnp.mean(a, axis=(1, 2), keepdims=True)
            scale = jnp.mean(scale, axis=(1, 2), keepdims=True)
            scale = jnp.minimum(
                scale / jnp.mean(scale, axis=-1, keepdims=True) * a * self.avg_scale,
                self.max_scale,
            )

        step = jnp.mean(conv(e, self.channels, 3, 1, nn.sigmoid, None))
        step = 126 * step + 2

        return (e, scale, step)


def step(x, p=16):
    return jnp.log((p - 1) * jnp.abs(x) + 1) / jnp.log(p) * jnp.sign(x)


def istep(x, p=16):
    return (p ** jnp.abs(x) - 1) / (p - 1) * jnp.sign(x)


def dct_mae(x, y, s=8):
    x = jnp.reshape(x, (x.shape[0], x.shape[1] // s, s, x.shape[2] // s, s, x.shape[3]))
    y = jnp.reshape(y, (y.shape[0], y.shape[1] // s, s, y.shape[2] // s, s, y.shape[3]))
    x = jnp.transpose(x, (0, 5, 1, 3, 2, 4))
    y = jnp.transpose(y, (0, 5, 1, 3, 2, 4))
    x = jnp.reshape(x, (-1, x.shape[4], x.shape[5]))
    y = jnp.reshape(y, (-1, y.shape[4], y.shape[5]))
    x = jax.vmap(dct)(x)
    y = jax.vmap(dct)(y)
    d = jnp.abs(x - y) * 0.1
    return jnp.mean(d)


def yuv_weighted_mae(x, y):
    x_luma = x[..., 0] * 0.29 + x[..., 1] * 0.587 + x[..., 2] * 0.114
    x_cb = x[..., 0] * -0.16874 + x[..., 1] * -0.33126 + x[..., 2] * 0.5
    x_cr = x[..., 0] * 0.5 + x[..., 1] * -0.41869 + x[..., 2] * -0.08131
    y_luma = y[..., 0] * 0.29 + y[..., 1] * 0.587 + y[..., 2] * 0.114
    y_cb = y[..., 0] * -0.16874 + y[..., 1] * -0.33126 + y[..., 2] * 0.5
    y_cr = y[..., 0] * 0.5 + y[..., 1] * -0.41869 + y[..., 2] * -0.08131
    d_luma = jnp.abs(x_luma - y_luma)
    d_luma = jnp.mean(d_luma)
    d_cb = jnp.abs(x_cb - y_cb)
    d_cb = jnp.mean(d_cb) * 0.5
    d_cr = jnp.abs(x_cr - y_cr)
    d_cr = jnp.mean(d_cr) * 0.5
    return d_luma + d_cb + d_cr


def yuv_mae(x, y):
    x_luma = x[..., 0] * 0.29 + x[..., 1] * 0.587 + x[..., 2] * 0.114
    x_cb = x[..., 0] * -0.16874 + x[..., 1] * -0.33126 + x[..., 2] * 0.5
    x_cr = x[..., 0] * 0.5 + x[..., 1] * -0.41869 + x[..., 2] * -0.08131
    y_luma = y[..., 0] * 0.29 + y[..., 1] * 0.587 + y[..., 2] * 0.114
    y_cb = y[..., 0] * -0.16874 + y[..., 1] * -0.33126 + y[..., 2] * 0.5
    y_cr = y[..., 0] * 0.5 + y[..., 1] * -0.41869 + y[..., 2] * -0.08131
    d_luma = jnp.abs(x_luma - y_luma)
    d_luma = jnp.mean(d_luma)
    d_cb = jnp.abs(x_cb - y_cb).mean()
    d_cr = jnp.abs(x_cr - y_cr).mean()
    return d_luma + d_cb + d_cr


class Decoder(nn.Module):
    num_blocks: int = 3

    @nn.compact
    def __call__(
        self,
        x,
        in_channels,
        mid_channels,
        out_channels,
        train,
        ksize,
        norm=None,
        dw=True,
    ):
        x = conv_transpose(x, in_channels, ksize, 2, mish, norm)
        x = res_blocks(x, in_channels, ksize, self.num_blocks, train, norm, dw)
        # x = conv_next_blocks(x, in_channels, ksize, self.num_blocks, train, norm, dw)
        x = conv_transpose(x, mid_channels, ksize, 2, mish, norm)
        x = res_blocks(x, mid_channels, ksize, self.num_blocks, train, norm, dw)
        # x = conv_next_blocks(x, mid_channels, ksize, self.num_blocks, train, norm, dw)
        x = conv(x, out_channels, ksize, 1, nn.sigmoid, None)
        return x


class Model(nn.Module):
    l1_encoder: Encoder
    l2_encoder: Encoder
    l3_encoder: Encoder
    l3_decoder: Decoder
    l2_decoder: Decoder
    l1_decoder: Decoder
    finetune: bool = False
    l1_step = 64
    l2_step = 16
    l3_step = 4

    @nn.compact
    def __call__(self, x, train, loss_scale=1.0, itr=0, tx=False):
        (l1, l1_res_scale, l1_step) = self.l1_encoder(x, train)
        (l2, l2_res_scale, l2_step) = self.l2_encoder(l1, train, True)
        (l3, l3_scale, l3_step) = self.l3_encoder(l2, train, True)
        l3 = l3 * 2 - 1

        if self.finetune:
            l1 = jax.lax.stop_gradient(l1)
            l2 = jax.lax.stop_gradient(l2)
            l3 = jax.lax.stop_gradient(l3)
            l1_res_scale = jax.lax.stop_gradient(l1_res_scale)
            l2_res_scale = jax.lax.stop_gradient(l2_res_scale)
            l3_scale = jax.lax.stop_gradient(l3_scale)

        # l1_cor = jnp.abs(
        # jnp.corrcoef(jnp.reshape(l1, [-1, l1.shape[-1]]), rowvar=False)
        # ).mean()

        l2_res_scale = jnp.reshape(l2_res_scale, (-1, l2_res_scale.shape[-1]))
        l1_res_scale = jnp.reshape(l1_res_scale, (-1, l1_res_scale.shape[-1]))

        scaled_quant_l3 = jnp.round(step(l3, self.l3_step) * l3_scale)
        quant_l3 = istep(scaled_quant_l3 / l3_scale, self.l3_step)
        l3_loss = jnp.abs(quant_l3).mean() * 0.1

        quant_l2_pred = self.l3_decoder(
            (quant_l3 * loss_scale + l3 * (1 - loss_scale))
            if train  # and itr % 2 == 0
            else quant_l3,
            self.l3_encoder.channels,
            self.l3_encoder.channels,
            self.l2_encoder.channels,
            train,
            3,
            layer_norm,
        )
        quant_l2_res = l2 - quant_l2_pred  # (N, H/64, W/64, 128)
        l2_res_loss = (
            +jnp.abs(quant_l2_res).mean() * 0.1
            + jnp.abs(jax.lax.stop_gradient(l2) - quant_l2_pred).mean() * 1.0
        )

        if tx:
            tx2 = LTX()
            itx2 = LTX()

            tx_quant_l2_res = tx2(quant_l2_res, quant_l2_res.shape[-1])
            itx_quant_l2_res = itx2(tx_quant_l2_res, tx_quant_l2_res.shape[-1])
            tx2_loss = jnp.abs(quant_l2_res - itx_quant_l2_res).mean()
            l2_res_loss += tx2_loss * 0.1

        if tx:
            quant_l2_res = jnp.reshape(tx_quant_l2_res, (-1, quant_l2_res.shape[-1]))
        else:
            quant_l2_res = jnp.reshape(quant_l2_res, (-1, quant_l2_res.shape[-1]))
        scaled_quant_l2_res = jnp.round(step(quant_l2_res, self.l2_step) * l2_res_scale)
        quant_l2_res = istep(scaled_quant_l2_res / l2_res_scale, self.l2_step)
        if tx:
            quant_l2_res = itx2(quant_l2_res, quant_l2_res.shape[-1])

        quant_l2_reconst = quant_l2_pred + jnp.reshape(
            quant_l2_res, quant_l2_pred.shape
        )

        quant_l1_pred = self.l2_decoder(
            (quant_l2_reconst * loss_scale + l2 * (1 - loss_scale))
            if train  # and itr % 2 == 0
            else quant_l2_reconst,
            self.l2_encoder.channels,
            self.l2_encoder.channels,
            self.l1_encoder.channels,
            train,
            3,
            layer_norm,
        )
        l2_reconst = self.l1_decoder(
            quant_l1_pred,
            self.l1_encoder.channels,
            self.l1_encoder.channels,
            3,
            train,
            5,
            layer_norm,
        )
        quant_l1_res = l1 - quant_l1_pred  # (N, H/16, W/16, 64)
        l1_res_loss = (
            jnp.abs(quant_l1_res).mean() * 0.2
            + jnp.abs(jax.lax.stop_gradient(l1) - quant_l1_pred).mean() * 1.0
        )

        if tx:
            tx1 = LTX()
            itx1 = LTX()

            tx_quant_l1_res = tx1(quant_l1_res, quant_l1_res.shape[-1])
            itx_quant_l1_res = itx1(tx_quant_l1_res, tx_quant_l1_res.shape[-1])
            tx1_loss = jnp.abs(quant_l1_res - itx_quant_l1_res).mean()
            l1_res_loss += tx1_loss * 0.1

        if tx:
            quant_l1_res = jnp.reshape(tx_quant_l1_res, (-1, quant_l1_res.shape[-1]))
        else:
            quant_l1_res = jnp.reshape(quant_l1_res, (-1, quant_l1_res.shape[-1]))
        scaled_quant_l1_res = jnp.round(step(quant_l1_res, self.l1_step) * l1_res_scale)
        quant_l1_res = istep(scaled_quant_l1_res / l1_res_scale, self.l1_step)
        if tx:
            quant_l1_res = itx1(quant_l1_res, quant_l1_res.shape[-1])

        quant_l1_reconst = quant_l1_pred + jnp.reshape(
            quant_l1_res, quant_l1_pred.shape
        )

        quant_reconst = self.l1_decoder(
            (quant_l1_reconst * loss_scale + l1 * (1 - loss_scale))
            if train  # and itr % 2 == 0
            else quant_l1_reconst,
            self.l1_encoder.channels,
            self.l1_encoder.channels,
            3,
            train,
            5,
            layer_norm,
        )
        e2e_reconst = self.l1_decoder(
            self.l2_decoder(
                quant_l2_pred,
                self.l2_encoder.channels,
                self.l2_encoder.channels,
                self.l1_encoder.channels,
                train,
                3,
                layer_norm,
            ),
            self.l1_encoder.channels,
            self.l1_encoder.channels,
            3,
            train,
            5,
            layer_norm,
        )

        # reconst_loss = (
        # jnp.abs(x - quant_reconst).mean() * 4.0
        # + jnp.abs(x - l2_reconst).mean() * 2.5
        # + jnp.abs(x - e2e_reconst).mean() * 1.5
        # ) * 2.0

        reconst_loss = (
            yuv_mae(x, quant_reconst) * 4.0
            + yuv_mae(x, l2_reconst) * 3.0
            + yuv_mae(x, e2e_reconst) * 2.0
        ) * 2.0

        l3_rate_estimator = RateEstimator(33, self.l3_encoder.channels)
        l1_res_rate_estimator = RateEstimator(33, self.l1_encoder.channels)
        l2_res_rate_estimator = RateEstimator(33, self.l2_encoder.channels)

        f_l3 = jax.vmap(lambda x: l3_rate_estimator(x, self.l3_encoder.channels))

        f_l1_res = jax.vmap(
            lambda x: l1_res_rate_estimator(x, self.l1_encoder.channels)
        )
        f_l2_res = jax.vmap(
            lambda x: l2_res_rate_estimator(x, self.l2_encoder.channels)
        )

        def estimate_bpp(q, scale, f, p=None):
            row_indices = jnp.reshape(jnp.arange(q.shape[1]), (1, -1, 1, 1))
            col_indices = jnp.reshape(jnp.arange(q.shape[2]), (1, 1, -1, 1))
            l_avail = jnp.ones(q.shape) * (col_indices > 0).astype(jnp.float32)
            a_avail = jnp.ones(q.shape) * (row_indices > 0).astype(jnp.float32)
            # ar_avail = (
            # jnp.ones(q.shape)
            # * (col_indices < q.shape[2] - 1).astype(jnp.float32)
            # * a_avail
            # )
            # al_avail = a_avail * l_avail
            # al_ref = jnp.roll(jnp.roll(q, 1, 2), 1, 1) * al_avail
            # ar_ref = jnp.roll(jnp.roll(q, -1, 2), 1, 1) * ar_avail
            l_ref = jnp.roll(q, 1, 2) * l_avail
            a_ref = jnp.roll(q, 1, 1) * a_avail
            ref = jnp.concatenate(
                [
                    # al_ref,
                    a_ref,
                    l_ref,
                    # ar_ref,
                    # al_avail[:, :, :, :1],
                    a_avail[:, :, :, :1],
                    l_avail[:, :, :, :1],
                    # ar_avail[:, :, :, :1],
                ],
                -1,
            )
            ref = jnp.reshape(ref, [-1, ref.shape[-1]])

            if p is not None:
                # al_pred = jnp.roll(jnp.roll(p, 1, 2), 1, 1) * al_avail
                # ar_pred = jnp.roll(jnp.roll(p, -1, 2), 1, 1) * ar_avail
                l_pred = jnp.roll(p, 1, 2) * l_avail
                a_pred = jnp.roll(p, 1, 1) * a_avail
                # pred = jnp.concatenate([al_pred, ar_pred, l_pred, a_pred, p], -1)
                pred = jnp.concatenate([l_pred, a_pred, p], -1)
                pred = jnp.reshape(pred, [-1, pred.shape[-1]])
                ref = jnp.concatenate([ref, pred], -1)

            q = jnp.reshape(q, (-1, q.shape[-1]))
            est = jnp.concatenate(
                [
                    q,  # N
                    ref,  # 4*N+4 or 9*N+4
                    jnp.reshape(scale, (-1, scale.shape[-1])),  # N
                ],
                -1,
            )

            bits = jnp.reshape(f(est), (x.shape[0], -1)).sum(-1)
            d = x.shape[1] * x.shape[2]
            bpp = bits / d

            return bpp

        l3_scale = jnp.reshape(l3_scale, (-1, l3_scale.shape[-1]))
        l3_bpp = estimate_bpp(scaled_quant_l3, l3_scale, f_l3)

        scaled_quant_l1_res = jnp.reshape(
            scaled_quant_l1_res, (*l1.shape[:3], self.l1_encoder.channels)
        )
        l1_res_bpp = estimate_bpp(
            scaled_quant_l1_res, l1_res_scale, f_l1_res, quant_l1_pred
        )

        scaled_quant_l2_res = step(quant_l2_res, self.l2_step) * l2_res_scale
        scaled_quant_l2_res = jnp.reshape(
            scaled_quant_l2_res, (*l2.shape[:3], self.l2_encoder.channels)
        )
        l2_res_bpp = estimate_bpp(
            scaled_quant_l2_res, l2_res_scale, f_l2_res, quant_l2_pred
        )

        bpp = l3_bpp + l1_res_bpp + l2_res_bpp

        return (
            quant_reconst,
            l3_bpp,
            l1_res_bpp,
            l2_res_bpp,
            bpp,
            reconst_loss,
            l1_res_loss,
            l2_res_loss,
            l3_loss,
            quant_reconst,
            e2e_reconst,
            l2_reconst,
        )
