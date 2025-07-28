import imageio
import numpy as np
from PIL import Image

from utils.typing import *


def save_images_to_gif(image_list, output_file, fps=15, verbose=False):
    """
    Save a list of images to a GIF file.

    Args:
        image_list (list of numpy arrays): A list of image frames (as numpy arrays).
        output_file (str): The path to the output GIF file.
        duration (float): The duration (in seconds) for each frame in the GIF.
    """
    if verbose:
        print(f"Save images to {output_file}")
    # Ensure that the output file ends with '.gif'
    if not output_file.endswith('.gif'):
        output_file += '.gif'
    # Save the images to a GIF
    with imageio.get_writer(output_file, mode='I', fps=fps, loop=0) as writer:
        for image in image_list:
            writer.append_data(image)


def slice_gif(output_file, gif_file, x_offset=0, x_size=None, y_offset=0, y_size=None):
    gif_reader = imageio.get_reader(gif_file)
    number_of_frames = gif_reader.get_length()
    out_gif = imageio.get_writer(output_file, mode='I', duration=0.08, loop=0)
    for frame_number in range(number_of_frames):
        new_image = gif_reader.get_next_data()
        if frame_number == 0:
            img_height, img_width = new_image.shape[:2]
            if x_size is None:
                x_size = min(img_width, img_width - x_offset)
            if y_size is None:
                y_size = min(img_height, img_height-y_offset)
        new_image = new_image[y_offset:y_offset+y_size, x_offset:x_offset+x_size]
        out_gif.append_data(new_image)
    out_gif.close()
    gif_reader.close()


def base_stack_gifs(
    stack_func: Callable[..., Any],
    output_file,
    result_gifs,
    input_image=None,
    is_img_prepend=True,
    duration=0.08,
    mode='RGB',
):
    """Stack gifs"""
    if type(input_image) is str:
        input_image = np.array(Image.open(input_image).convert(mode))
    elif isinstance(input_image, Image.Image):
        input_image = input_image.convert(mode)
    elif input_image is None:
        pass
    else:
        raise NotImplementedError(f"Do not support input_image type {type(input_image)}")

    gif_reader_list = [imageio.get_reader(gif_path) for gif_path in result_gifs]
    number_of_frames = min([x.get_length() for x in gif_reader_list])
    out_gif = imageio.get_writer(output_file, mode='I', duration=duration, loop=0)
    for frame_number in range(number_of_frames-1):
        new_image = [x.get_next_data() for x in gif_reader_list]
        new_image = [np.array(Image.fromarray(np.array(cur_img)).convert(mode)) for cur_img in new_image]
        if input_image is not None:
            if is_img_prepend:
                new_image = [input_image] + new_image
            else:
                new_image = new_image + [input_image]
        new_image = stack_func(new_image)
        out_gif.append_data(new_image)
    out_gif.close()
    for gif_reader in gif_reader_list:
        gif_reader.close()


def hstack_gifs(
    output_file,
    result_gifs,
    input_image=None,
    is_img_prepend=True,
    duration=0.08,
    mode='RGB'
):
    """Stack gifs in sequence horizontally (column wise)."""
    return base_stack_gifs(
        stack_func=np.hstack,
        output_file=output_file,
        result_gifs=result_gifs,
        input_image=input_image,
        is_img_prepend=is_img_prepend,
        duration=duration,
        mode=mode,
    )


def vstack_gifs(
    output_file,
    result_gifs,
    input_image=None,
    is_img_prepend=False,
    duration=0.08,
    mode='RGB'
):
    """Stack gifs in sequence vertically (row wise)."""
    return base_stack_gifs(
        stack_func=np.vstack,
        output_file=output_file,
        result_gifs=result_gifs,
        input_image=input_image,
        is_img_prepend=is_img_prepend,
        duration=duration,
        mode=mode,
    )
