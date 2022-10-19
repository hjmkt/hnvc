import jax
import optax
import argparse
import os
import numpy as np
import glob
import tensorflow as tf
import functools
import pillow_avif
import cv2
from PIL import Image
from io import BytesIO
from sklearn.model_selection import train_test_split
from flax.training import train_state
from flax.metrics import tensorboard
from tqdm import tqdm
from flax.training import checkpoints
from util import MetricsLogger
from clu import parameter_overview
from network import *


TRAIN_BATCH_SIZE = 256
TEST_BATCH_SIZE = 256
TRAIN_H, TRAIN_W = 64, 64
TEST_H, TEST_W = 64, 64


def path_to_image(image, training=True):
    th, tw = (TRAIN_H, TRAIN_W) if training else (TEST_H, TEST_W)
    image = tf.io.read_file(image)
    image = tf.image.decode_image(image)
    shape = tf.shape(image)
    if training:
        if shape[-1] == 1:
            image = tf.image.random_crop(image, (th, tw, 1))
        else:
            image = tf.image.random_crop(image, (th, tw, 3))
    else:
        oy = (shape[0] - th) // 2
        ox = (shape[1] - tw) // 2
        image = image[oy : oy + th, ox : ox + tw]
    if shape[-1] == 1:
        image = tf.tile(image, [1, 1, 3])
    image = tf.reshape(image, (th, tw, 3))
    image = tf.cast(image, tf.float32) / 255.0
    return image


def train_path_to_image(image):
    return path_to_image(image, True)


def test_path_to_image(image):
    return path_to_image(image, False)


@tf.function
def preprocess(images, training=True):
    image = tf.map_fn(
        train_path_to_image if training else test_path_to_image,
        images,
        dtype=tf.float32,
        parallel_iterations=16,
    )
    if training:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, 0.15)
        image = tf.image.random_contrast(image, 0.85, 1.15)
    return image


@tf.function
def train_preprocess(img):
    return preprocess(img, True)


@tf.function
def test_preprocess(img):
    return preprocess(img, False)


def create_tf_dataset(img_path):
    img_files = glob.glob(os.path.join(img_path, "*"))
    img_files.sort()
    train_images, test_images = train_test_split(
        img_files, test_size=0.1, random_state=42
    )
    train_images = train_images[
        : len(train_images) // TRAIN_BATCH_SIZE * TRAIN_BATCH_SIZE
    ]
    test_images = test_images[: len(test_images) // TEST_BATCH_SIZE * TEST_BATCH_SIZE]

    train_ds = tf.data.Dataset.from_tensor_slices(train_images)
    train_ds = (
        train_ds.shuffle(buffer_size=200000)
        .batch(TRAIN_BATCH_SIZE)
        .map(train_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
        .map(lambda x: (x, x))
    )
    test_ds = tf.data.Dataset.from_tensor_slices(test_images)
    test_ds = (
        test_ds.batch(TEST_BATCH_SIZE)
        .map(test_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
        .map(lambda x: (x, x))
    )
    return train_ds, test_ds, len(train_images), len(test_images)


@functools.partial(jax.jit, static_argnums=3)
def training_step(model_state, images, loss_scale, learning_rate_fn, itr):
    def loss_fn(
        model_params,
        is_training,
    ):
        (
            quant_reconst,
            l3_bpp,
            l1_res_bpp,
            l2_res_bpp,
            bpp,
            reconst_loss,
            l1_res_loss,
            l2_res_loss,
            l3_loss,
            l1_reconst,
            e2e_reconst,
            l2_reconst,
        ) = model_state.apply_fn({"params": model_params}, images, True, loss_scale)
        res_loss = l1_res_loss + l2_res_loss + l3_loss
        loss = reconst_loss + res_loss + bpp.mean() * 0.5
        return loss, (
            quant_reconst,
            l3_bpp,
            l1_res_bpp,
            l2_res_bpp,
            bpp,
            l1_res_loss,
            l2_res_loss,
            l1_reconst,
            e2e_reconst,
            l2_reconst,
        )

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (
        loss,
        (
            quant_reconst,
            l3_bpp,
            l1_res_bpp,
            l2_res_bpp,
            bpp,
            l1_res_loss,
            l2_res_loss,
            l1_reconst,
            e2e_reconst,
            l2_reconst,
        ),
    ), grads = grad_fn(
        model_state.params,
        True,
    )
    model_state = model_state.apply_gradients(grads=grads)
    lr = learning_rate_fn(model_state.step)
    return (
        loss,
        model_state,
        quant_reconst,
        l3_bpp,
        l1_res_bpp,
        l2_res_bpp,
        bpp,
        l1_res_loss,
        l2_res_loss,
        l1_reconst,
        e2e_reconst,
        l2_reconst,
        lr,
    )


@jax.jit
def test_step(model_state, images):
    def loss_fn(model_params, is_training):
        (
            quant_reconst,
            l3_bpp,
            l1_res_bpp,
            l2_res_bpp,
            bpp,
            reconst_loss,
            l1_res_loss,
            l2_res_loss,
            l3_loss,
            l1_reconst,
            e2e_reconst,
            l2_reconst,
        ) = model_state.apply_fn(
            {"params": model_params},
            images,
            False,
        )
        res_loss = l1_res_loss + l2_res_loss + l3_loss
        loss = reconst_loss + res_loss + bpp.mean() * 0.5
        return loss, (
            quant_reconst,
            l3_bpp,
            l1_res_bpp,
            l2_res_bpp,
            bpp,
            l1_res_loss,
            l2_res_loss,
            l1_reconst,
            e2e_reconst,
            l2_reconst,
        )

    loss, (
        quant_reconst,
        l3_bpp,
        l1_res_bpp,
        l2_res_bpp,
        bpp,
        l1_res_loss,
        l2_res_loss,
        l1_reconst,
        e2e_reconst,
        l2_reconst,
    ) = loss_fn(model_state.params, False)
    return (
        loss,
        quant_reconst,
        l3_bpp,
        l1_res_bpp,
        l2_res_bpp,
        bpp,
        l1_res_loss,
        l2_res_loss,
        l1_reconst,
        e2e_reconst,
        l2_reconst,
    )


def create_train_state(learning_rate_fn, finetune=False):
    l1_encoder = Encoder(16, 24, 5, 5, 1.0, 32, 3)
    l2_encoder = Encoder(48, 48, 5, 3, 3, 32, 2)
    l3_encoder = Encoder(96, 96, 3, 3, 8, 32, 1)
    l3_decoder = Decoder(1)
    l2_decoder = Decoder(2)
    l1_decoder = Decoder(3)
    model = Model(
        l1_encoder, l2_encoder, l3_encoder, l3_decoder, l2_decoder, l1_decoder, finetune
    )
    model_vars = model.init(
        jax.random.PRNGKey(0), jnp.ones((1, TRAIN_H, TRAIN_W, 3)), True
    )
    model_params = model_vars["params"]
    model_tx = optax.adabelief(learning_rate_fn)
    model_state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=model_params,
        tx=model_tx,
    )
    return model_state


def create_learning_rate_fn(
    base_learning_rate, num_epochs, warmup_epochs, steps_per_epoch
):
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=base_learning_rate,
        transition_steps=warmup_epochs * steps_per_epoch,
    )
    cosine_epochs = max(num_epochs - warmup_epochs, 1)
    cosine_fn = optax.cosine_decay_schedule(
        init_value=base_learning_rate, decay_steps=cosine_epochs * steps_per_epoch
    )
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[warmup_epochs * steps_per_epoch],
    )
    return schedule_fn


def create_loss_scale_fn(min_scale, max_scale, scale_epochs, steps_per_epoch):
    scale_fn = optax.linear_schedule(
        init_value=min_scale,
        end_value=max_scale,
        transition_steps=scale_epochs * steps_per_epoch,
    )
    return scale_fn


def train_and_evaluate(
    dataset_path,
    epochs,
    tb_log_dir,
    ckpt_dir,
    finetune=False,
    lr=1e-3,
    min_ls=0.5,
    max_ls=0.8,
):
    train_ds, test_ds, n_train, n_test = create_tf_dataset(dataset_path)
    train_ds_size = len(train_ds)
    summary_writer = tensorboard.SummaryWriter(tb_log_dir)
    learning_rate_fn = create_learning_rate_fn(
        (lr / 2) if finetune else lr, epochs, 2, train_ds_size
    )
    loss_scale_fn = create_loss_scale_fn(min_ls, max_ls, epochs // 2, train_ds_size)
    model_state = create_train_state(learning_rate_fn, finetune)

    restored_vars = checkpoints.restore_checkpoint(
        ckpt_dir,
        {"model_state": model_state},
    )
    model_state = restored_vars["model_state"]
    model_state.__dict__["step"] = 0
    # print(parameter_overview.get_parameter_overview(model_state.params))

    best_test_loss = 1e10
    train_metrics_logger = MetricsLogger(
        [
            "loss",
            "b",
            "r1",
            "r2",
            "b1",
            "b2",
            "b3",
            # "psnr",
            "s",
            "s2",
            "s3",
            "lr",
            "w",
        ],
        train=True,
    )
    test_metrics_logger = MetricsLogger(
        [
            "loss",
            "b",
            "r1",
            "r2",
            "b1",
            "b2",
            "b3",
            # "psnr",
            "s",
            "s2",
            "s3",
        ],
        train=False,
    )

    for epoch in range(epochs):
        bar = tqdm(total=train_ds_size)
        bar.set_description(f"Train #{epoch}")
        for itr, batch in enumerate(train_ds.as_numpy_iterator()):
            loss_scale = loss_scale_fn(model_state.step)
            (
                loss,
                model_state,
                decoded,
                l3_bpp,
                l1_res_bpp,
                l2_res_bpp,
                bpp,
                l1_res_loss,
                l2_res_loss,
                l1_reconst,
                e2e_reconst,
                l2_reconst,
                lr,
            ) = training_step(model_state, batch[0], loss_scale, learning_rate_fn, itr)
            original_yuv = tf.cast(
                tf.clip_by_value(
                    tf.math.round(tf.image.rgb_to_yuv(batch[0]) * 255), 0, 255
                ),
                tf.uint8,
            )
            decoded_yuv = tf.cast(
                tf.clip_by_value(
                    tf.math.round(tf.image.rgb_to_yuv(decoded) * 255), 0, 255
                ),
                tf.uint8,
            )
            l2_yuv = tf.cast(
                tf.clip_by_value(
                    tf.math.round(tf.image.rgb_to_yuv(l2_reconst) * 255), 0, 255
                ),
                tf.uint8,
            )
            l3_yuv = tf.cast(
                tf.clip_by_value(
                    tf.math.round(tf.image.rgb_to_yuv(e2e_reconst) * 255), 0, 255
                ),
                tf.uint8,
            )
            # psnr = tf.reduce_mean(tf.image.psnr(original_rgb, decoded_rgb, 255))
            ssim = tf.reduce_mean(tf.image.ssim(original_yuv, decoded_yuv, 255))
            ssim2 = tf.image.ssim(original_yuv, l2_yuv, 255)
            ssim3 = tf.image.ssim(original_yuv, l3_yuv, 255)
            train_metrics_logger.update(
                epoch,
                {
                    "loss": loss,
                    "b": bpp.mean(),
                    "b1": l1_res_bpp.mean(),
                    "b2": l2_res_bpp.mean(),
                    "b3": l3_bpp.mean(),
                    "r1": l1_res_loss,
                    "r2": l2_res_loss,
                    # "psnr": psnr,
                    "s": ssim,
                    "s2": tf.reduce_mean(ssim2),
                    "s3": tf.reduce_mean(ssim3),
                    "lr": lr,
                    "w": loss_scale,
                },
                bar,
            )
            if itr % 100 == 0:
                summary_image = jnp.concatenate(
                    [
                        batch[0] * 255.0,
                        decoded * 255.0,
                        # l1_reconst * 255.0,
                        l2_reconst * 255.0,
                        e2e_reconst * 255.0,
                    ],
                    axis=2,
                )
                summary_writer.image(
                    "train/image",
                    jnp.clip(jnp.round(summary_image), 0, 255).astype(jnp.uint8),
                    epoch,
                    16,
                )
        train_metrics_logger.set_summary(summary_writer)

        test_ds_size = len(test_ds)
        bar = tqdm(total=test_ds_size)
        bar.set_description(f"Test #{epoch}")
        for itr, batch in enumerate(test_ds.as_numpy_iterator()):
            (
                loss,
                decoded,
                l3_bpp,
                l1_res_bpp,
                l2_res_bpp,
                bpp,
                l1_res_loss,
                l2_res_loss,
                l1_reconst,
                e2e_reconst,
                l2_reconst,
            ) = test_step(
                model_state,
                batch[0],
            )
            original_yuv = tf.cast(
                tf.clip_by_value(
                    tf.math.round(tf.image.rgb_to_yuv(batch[0]) * 255), 0, 255
                ),
                tf.uint8,
            )
            decoded_yuv = tf.cast(
                tf.clip_by_value(
                    tf.math.round(tf.image.rgb_to_yuv(decoded) * 255), 0, 255
                ),
                tf.uint8,
            )
            l2_yuv = tf.cast(
                tf.clip_by_value(
                    tf.math.round(tf.image.rgb_to_yuv(l2_reconst) * 255), 0, 255
                ),
                tf.uint8,
            )
            l3_yuv = tf.cast(
                tf.clip_by_value(
                    tf.math.round(tf.image.rgb_to_yuv(e2e_reconst) * 255), 0, 255
                ),
                tf.uint8,
            )

            # psnr = tf.image.psnr(original_yuv, decoded_yuv, 255)
            ssim = tf.image.ssim(original_yuv, decoded_yuv, 255)
            ssim2 = tf.image.ssim(original_yuv, l2_yuv, 255)
            ssim3 = tf.image.ssim(original_yuv, l3_yuv, 255)

            test_metrics_logger.update(
                epoch,
                {
                    "loss": loss,
                    "b": bpp.mean(),
                    "b1": l1_res_bpp.mean(),
                    "b2": l2_res_bpp.mean(),
                    "b3": l3_bpp.mean(),
                    "r1": l1_res_loss,
                    "r2": l2_res_loss,
                    # "psnr": tf.reduce_mean(psnr),
                    "s": tf.reduce_mean(ssim),
                    "s2": tf.reduce_mean(ssim2),
                    "s3": tf.reduce_mean(ssim3),
                },
                bar,
            )
            summary_image_size = (TEST_H, TEST_W)
            if itr == 0:
                num_images = 64
                if itr == 0 and epoch == 0:
                    qualities = [37, 25, 13, 1]
                    avif_images = []
                    for quality in qualities:
                        pil_images = []
                        for b in range(TEST_BATCH_SIZE):
                            pil_image = tf.cast(
                                tf.clip_by_value(
                                    tf.math.round(batch[0][b] * 255.0), 0, 255
                                ),
                                tf.uint8,
                            ).numpy()
                            pil_image = Image.fromarray(pil_image)
                            buf = BytesIO()
                            pil_image.save(buf, format="AVIF", quality=quality, speed=2)
                            avif_bpp = buf.getbuffer().nbytes * 8 / TEST_W / TEST_H
                            pil_image = Image.open(buf)
                            avif_image = np.array(pil_image, np.float32)
                            avif_yuv = tf.clip_by_value(
                                tf.image.rgb_to_yuv(avif_image / 255) * 255, 0, 255
                            )
                            avif_yuv = tf.cast(avif_yuv, tf.uint8)
                            avif_ssim = tf.reduce_mean(
                                tf.image.ssim(original_yuv[b], avif_yuv, 255)
                            ).numpy()
                            ann_image = np.full(
                                (summary_image_size[0] // 4, summary_image_size[1], 3),
                                255,
                                dtype=np.uint8,
                            )
                            cv2.putText(
                                ann_image,
                                text=f"{avif_ssim:.3f}@{avif_bpp:.3f}",
                                org=(
                                    1,
                                    TEST_H // 7,
                                ),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=TEST_W / 256,
                                color=(0, 0, 0),
                                thickness=1,
                                lineType=cv2.LINE_AA,
                            )
                            avif_image = tf.concat([avif_image, ann_image], 0)
                            pil_images.append(avif_image)
                        avif_image = tf.stack(pil_images, 0)
                        avif_images.append(avif_image)
                    avif_images = tf.concat(avif_images, 2)

                ann_image = np.full(
                    (
                        TEST_BATCH_SIZE,
                        summary_image_size[0] // 4,
                        summary_image_size[1] * 4,
                        3,
                    ),
                    255,
                    dtype=np.uint8,
                )
                for b in range(TEST_BATCH_SIZE):
                    cv2.putText(
                        ann_image[b],
                        text=f"{ssim[b]:.3f}@{bpp[b]:.3f}",
                        org=(summary_image_size[1] + 1, TEST_H // 7),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=TEST_W / 256,
                        color=(0, 0, 0),
                        thickness=1,
                        lineType=cv2.LINE_AA,
                    )
                    cv2.putText(
                        ann_image[b],
                        text=f"{ssim2[b]:.3f}@{l2_res_bpp[b]+l3_bpp[b]:.3f}",
                        org=(summary_image_size[1] * 2 + 1, TEST_H // 7),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=TEST_W / 256,
                        color=(0, 0, 0),
                        thickness=1,
                        lineType=cv2.LINE_AA,
                    )
                    cv2.putText(
                        ann_image[b],
                        text=f"{ssim3[b]:.3f}@{l3_bpp[b]:.3f}",
                        org=(summary_image_size[1] * 3 + 1, TEST_H // 7),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=TEST_W / 256,
                        color=(0, 0, 0),
                        thickness=1,
                        lineType=cv2.LINE_AA,
                    )
                summary_image = tf.concat(
                    [
                        batch[0] * 255.0,
                        decoded * 255.0,
                        # l1_reconst * 255.0,
                        l2_reconst * 255.0,
                        e2e_reconst * 255.0,
                    ],
                    axis=2,
                )
                summary_image = tf.concat([summary_image, ann_image, avif_images], 1)
                summary_writer.image(
                    "test/image",
                    tf.cast(
                        tf.clip_by_value(tf.math.round(summary_image), 0, 255), tf.uint8
                    ),
                    epoch,
                    num_images,
                )
        test_metrics_logger.set_summary(summary_writer)

        loss_test_avg = test_metrics_logger.get("loss")
        ssim_test_avg = test_metrics_logger.get("s")
        bpp_test_avg = test_metrics_logger.get("b")
        checkpoints.save_checkpoint(
            f"{ckpt_dir}/e{epoch}_l{loss_test_avg}_b{bpp_test_avg}_s{ssim_test_avg}",
            {"model_state": model_state},
            epoch,
            overwrite=True,
        )
        if loss_test_avg < best_test_loss:
            best_test_loss = loss_test_avg
            checkpoints.save_checkpoint(
                ckpt_dir,
                {"model_state": model_state},
                epoch,
                overwrite=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HNVC")
    parser.add_argument("--path", type=str, help="Path to training/test images")
    parser.add_argument("--logdir", type=str, help="Path to TensorBoard logs")
    parser.add_argument("--ckpt", type=str, help="Path to checkpoint files")
    parser.add_argument(
        "--epochs", type=int, default=100, help="The max number of epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--minls", type=float, default=0.5, help="Min loss scale")
    parser.add_argument("--maxls", type=float, default=0.8, help="Max loss scale")
    parser.add_argument("--bs", type=int, default=256, help="Batch size")
    parser.add_argument("--width", type=int, default=64, help="Image width")
    parser.add_argument("--height", type=int, default=64, help="Image height")
    parser.add_argument("--finetune", action="store_true", help="Finetune or not")
    args = parser.parse_args()
    TRAIN_BATCH_SIZE = args.bs
    TEST_BATCH_SIZE = args.bs
    TRAIN_H, TRAIN_W = args.height, args.width
    TEST_H, TEST_W = args.height, args.width
    train_and_evaluate(
        args.path,
        args.epochs,
        args.logdir,
        args.ckpt,
        args.finetune,
        args.lr,
        args.minls,
        args.maxls,
    )
