import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision.utils


def stack_images(*images):
    """This function stacks images along the third axis.
    This is useful for combining e.g. rgb color channels or color and alpha channels.

    Parameters
    ----------
    *images: numpy.ndarray
        Images to be stacked.

    Returns
    -------
    image: numpy.ndarray
        Stacked images as numpy.ndarray
    """
    images = [
        (image if len(image.shape) == 3 else image[:, :, np.newaxis])
        for image in images
    ]
    return np.concatenate(images, axis=2)


def make_grid(images, nx=None, ny=None, dtype=None):
    """Plots a grid of images.

    Parameters
    ----------
    images : list of numpy.ndarray
        List of images to plot
    nx: int
        Number of rows
    ny: int
        Number of columns
    dtype: type
        Data type of output array

    Returns
    -------
    grid: numpy.ndarray
       Grid of images with datatype `dtype`
    """
    for image in images:
        if image is not None:
            assert image.dtype in [np.float32, np.float64]

    n = len(images)

    if n == 0:
        return

    if nx is None and ny is None:
        nx = int(np.ceil(np.sqrt(n)))
        ny = (n + nx - 1) // nx

    elif ny is None:
        ny = (n + nx - 1) // nx

    elif nx is None:
        nx = (n + ny - 1) // ny

    shapes = [image.shape for image in images if image is not None]

    h = max(shape[0] for shape in shapes)
    w = max(shape[1] for shape in shapes)
    d = max([shape[2] for shape in shapes if len(shape) > 2], default=1)

    if d > 1:
        for i, image in enumerate(images):
            if image is not None:
                if len(image.shape) == 2:
                    image = image[:, :, np.newaxis]

                if image.shape[2] == 1:
                    image = np.concatenate([image] * d, axis=2)

                if image.shape[2] == 3 and d == 4:
                    image = stack_images(
                        image, np.ones(image.shape[:2], dtype=image.dtype)
                    )

                images[i] = image

    if dtype is None:
        dtype = next(image.dtype for image in images if image is not None)

    result = np.zeros((h * ny, w * nx, d), dtype=dtype)

    for y in range(ny):
        for x in range(nx):
            i = x + y * nx

            if i >= len(images):
                break

            image = images[i]

            if image is not None:
                image = image.reshape(image.shape[0], image.shape[1], -1)

                result[
                    y * h : y * h + image.shape[0], x * w : x * w + image.shape[1]
                ] = image

    if result.shape[2] == 1:
        result = result[:, :, 0]

    return result


def tensor2im(image_tensor: torch.Tensor, data_range=(-1.0, 1.0), max_samples=16, nrow=4):
    n_img = min(image_tensor.shape[0], max_samples)
    image_tensor = image_tensor[:n_img]

    if image_tensor.shape[1] == 1:
        image_tensor = image_tensor.repeat(1, 3, 1, 1)

    collage = torchvision.utils.make_grid(image_tensor, nrow=nrow)

    image_numpy = collage.cpu().float().numpy()
    min_value, max_value = data_range
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) - min_value) / (max_value - min_value)
    return (image_numpy * 255.).astype(np.uint8)

