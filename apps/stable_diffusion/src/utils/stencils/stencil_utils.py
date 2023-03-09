import cv2
import numpy as np
from PIL import Image
import torch
from apps.stable_diffusion.src.utils.stencils import (
    CannyDetector,
    OpenposeDetector,
)

stencil = {}


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(
        input_image,
        (W, H),
        interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA,
    )
    return img


def controlnet_hint_shaping(
    controlnet_hint, height, width, dtype, num_images_per_prompt=1
):
    channels = 3
    if isinstance(controlnet_hint, torch.Tensor):
        # torch.Tensor: acceptble shape are any of chw, bchw(b==1) or bchw(b==num_images_per_prompt)
        shape_chw = (channels, height, width)
        shape_bchw = (1, channels, height, width)
        shape_nchw = (num_images_per_prompt, channels, height, width)
        if controlnet_hint.shape in [shape_chw, shape_bchw, shape_nchw]:
            controlnet_hint = controlnet_hint.to(
                dtype=dtype, device=torch.device("cpu")
            )
            if controlnet_hint.shape != shape_nchw:
                controlnet_hint = controlnet_hint.repeat(
                    num_images_per_prompt, 1, 1, 1
                )
            return controlnet_hint
        else:
            raise ValueError(
                f"Acceptble shape of `stencil` are any of ({channels}, {height}, {width}),"
                + f" (1, {channels}, {height}, {width}) or ({num_images_per_prompt}, "
                + f"{channels}, {height}, {width}) but is {controlnet_hint.shape}"
            )
    elif isinstance(controlnet_hint, np.ndarray):
        # np.ndarray: acceptable shape is any of hw, hwc, bhwc(b==1) or bhwc(b==num_images_per_promot)
        # hwc is opencv compatible image format. Color channel must be BGR Format.
        if controlnet_hint.shape == (height, width):
            controlnet_hint = np.repeat(
                controlnet_hint[:, :, np.newaxis], channels, axis=2
            )  # hw -> hwc(c==3)
        shape_hwc = (height, width, channels)
        shape_bhwc = (1, height, width, channels)
        shape_nhwc = (num_images_per_prompt, height, width, channels)
        if controlnet_hint.shape in [shape_hwc, shape_bhwc, shape_nhwc]:
            controlnet_hint = torch.from_numpy(controlnet_hint.copy())
            controlnet_hint = controlnet_hint.to(
                dtype=torch.half, device=torch.device("cpu")
            )
            controlnet_hint /= 255.0
            if controlnet_hint.shape != shape_nhwc:
                controlnet_hint = controlnet_hint.repeat(
                    num_images_per_prompt, 1, 1, 1
                )
            controlnet_hint = controlnet_hint.permute(
                0, 3, 1, 2
            )  # b h w c -> b c h w
            return controlnet_hint
        else:
            raise ValueError(
                f"Acceptble shape of `stencil` are any of ({width}, {channels}), "
                + f"({height}, {width}, {channels}), "
                + f"(1, {height}, {width}, {channels}) or "
                + f"({num_images_per_prompt}, {channels}, {height}, {width}) but is {controlnet_hint.shape}"
            )
    elif isinstance(controlnet_hint, Image.Image):
        if controlnet_hint.size == (width, height):
            controlnet_hint = controlnet_hint.convert(
                "RGB"
            )  # make sure 3 channel RGB format
            controlnet_hint = np.array(controlnet_hint)  # to numpy
            controlnet_hint = controlnet_hint[:, :, ::-1]  # RGB -> BGR
            return controlnet_hint_shaping(
                controlnet_hint, height, width, num_images_per_prompt
            )
        else:
            raise ValueError(
                f"Acceptable image size of `stencil` is ({width}, {height}) but is {controlnet_hint.size}"
            )
    else:
        raise ValueError(
            f"Acceptable type of `stencil` are any of torch.Tensor, np.ndarray, PIL.Image.Image but is {type(controlnet_hint)}"
        )


def controlnet_hint_conversion(
    image, use_stencil, height, width, dtype, num_images_per_prompt=1
):
    # controlnet_hint = controlnet_hint_shaping(
    #     image, height, width, dtype, num_images_per_prompt
    # )
    match use_stencil:
        case "canny":
            print("Detecting edge with canny")
            controlnet_hint = hint_canny(image, width)
        case "openpose":
            print("Detecting human pose")
            controlnet_hint = hint_openpose(image, width)
        case _:
            return None
    controlnet_hint = controlnet_hint_shaping(
        image, height, width, dtype, num_images_per_prompt
    )
    # controlnet_hint = torch.from_numpy(controlnet_hint.copy())
    # controlnet_hint = controlnet_hint.to(
    #             dtype=dtype, device=torch.device("cpu")
    #         )
    # controlnet_hint /= 255.0
    # controlnet_hint = controlnet_hint.permute(
    #             0, 3, 1, 2
    #         )  # b h w c -> b c h w
    return controlnet_hint

stencil_to_model_id_map = {
    "canny": "lllyasviel/sd-controlnet-canny",
    "depth": "lllyasviel/sd-controlnet-depth",
    "hed": "lllyasviel/sd-controlnet-hed",
    "mlsd": "lllyasviel/sd-controlnet-mlsd",
    "normal": "lllyasviel/sd-controlnet-normal",
    "openpose": "lllyasviel/sd-controlnet-openpose",
    "scribble": "lllyasviel/sd-controlnet-scribble",
    "seg": "lllyasviel/sd-controlnet-seg",
}

def get_stencil_model_id(use_stencil):
    if use_stencil in stencil_to_model_id_map:
        print("Returning ModelID: ", str(stencil_to_model_id_map[use_stencil]))
        return stencil_to_model_id_map[use_stencil]
    return None

# Stencil 1. Canny.
def hint_canny(
    image: Image.Image,
    width=512,
    height=512,
    low_threshold=100,
    high_threshold=200,
):
    with torch.no_grad():
        input_image = np.array(image)
        image_resolution = width

        img = resize_image(HWC3(input_image), image_resolution)

        if not "canny" in stencil:
            stencil["canny"] = CannyDetector()
        detected_map = stencil["canny"](img, low_threshold, high_threshold)
        detected_map = HWC3(detected_map)
        return detected_map

# Stencil 2. OpenPose.
def hint_openpose(
    image: Image.Image,
    width=512,
    height=512,
    detect_resolution=512,
):
    with torch.no_grad():
        input_image = np.array(image)
        image_resolution = width

        input_image = HWC3(input_image)

        if not "openpose" in stencil:
            stencil["openpose"] = OpenposeDetector()

        detected_map, _ = stencil["openpose"](
            resize_image(input_image, detect_resolution))
        detected_map = HWC3(detected_map)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape
        detected_map = cv2.resize(
            detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        return Image.fromarray(detected_map)
