# Use code from https://github.com/hendrycks/robustness

import collections
from io import BytesIO
from PIL import Image as PILImage
import time

import cv2
from loguru import logger
import numpy as np
import skimage as sk
from skimage.filters import gaussian
from scipy.ndimage import zoom as scizoom
from scipy.ndimage.interpolation import map_coordinates
from wand.image import Image as WandImage
from wand.api import library as wandlibrary


# /////////////// Distortion Helpers ///////////////


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


# Extend wand.image.Image class to include method signature
class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)


# modification of https://github.com/FLHerne/mapgen/blob/master/diamondsquare.py
def plasma_fractal(image_size, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    mapsize = 2 ** (image_size - 1).bit_length()
    assert mapsize & (mapsize - 1) == 0
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
        calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[
            stepsize // 2 : mapsize : stepsize, stepsize // 2 : mapsize : stepsize
        ] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
        calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[
            stepsize // 2 : mapsize : stepsize, stepsize // 2 : mapsize : stepsize
        ]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2 : mapsize : stepsize] = wibbledmean(
            ltsum
        )
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2 : mapsize : stepsize, 0:mapsize:stepsize] = wibbledmean(
            ttsum
        )

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / zoom_factor))

    top = (h - ch) // 2
    img = scizoom(
        img[top : top + ch, top : top + ch], (zoom_factor, zoom_factor, 1), order=1
    )
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2

    return img[trim_top : trim_top + h, trim_top : trim_top + h]


# /////////////// End Distortion Helpers ///////////////


# /////////////// Distortions ///////////////


def gaussian_noise(x, severity_params, image_size):
    x = np.array(x) / 255.0
    return (
        np.clip(x + np.random.normal(size=x.shape, scale=severity_params), 0, 1) * 255
    )


def shot_noise(x, severity_params, image_size):
    x = np.array(x) / 255.0
    return np.clip(np.random.poisson(x * severity_params) / severity_params, 0, 1) * 255


def impulse_noise(x, severity_params, image_size):
    x = sk.util.random_noise(np.array(x) / 255.0, mode="s&p", amount=severity_params)
    return np.clip(x, 0, 1) * 255


def speckle_noise(x, severity_params, image_size):
    x = np.array(x) / 255.0
    return (
        np.clip(x + x * np.random.normal(size=x.shape, scale=severity_params), 0, 1)
        * 255
    )


def gaussian_blur(x, severity_params, image_size):
    x = gaussian(np.array(x) / 255.0, sigma=severity_params, multichannel=True)
    return np.clip(x, 0, 1) * 255


def glass_blur(x, severity_params, image_size):
    x = np.uint8(
        gaussian(np.array(x) / 255.0, sigma=severity_params[0], multichannel=True) * 255
    )

    # locally shuffle pixels
    for i in range(severity_params[2]):
        for h in range(image_size - severity_params[1], severity_params[1], -1):
            for w in range(image_size - severity_params[1], severity_params[1], -1):
                dx, dy = np.random.randint(
                    -severity_params[1], severity_params[1], size=(2,)
                )
                h_prime, w_prime = h + dy, w + dx
                # swap
                x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

    return (
        np.clip(gaussian(x / 255.0, sigma=severity_params[0], multichannel=True), 0, 1)
        * 255
    )


def defocus_blur(x, severity_params, image_size):
    x = np.array(x) / 255.0
    kernel = disk(radius=severity_params[0], alias_blur=severity_params[1])

    channels = []
    for d in range(3):
        channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
    channels = np.array(channels).transpose(
        (1, 2, 0)
    )  # 3ximage_sizeximage_size -> image_sizeximage_sizex3

    return np.clip(channels, 0, 1) * 255


def motion_blur(x, severity_params, image_size):
    output = BytesIO()
    x.save(output, format="PNG")
    x = MotionImage(blob=output.getvalue())

    x.motion_blur(
        radius=severity_params[0],
        sigma=severity_params[1],
        angle=np.random.uniform(-45, 45),
    )

    x = cv2.imdecode(np.fromstring(x.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED)

    if x.shape != (image_size, image_size):
        return np.clip(x[..., [2, 1, 0]], 0, 255)  # BGR to RGB
    else:  # greyscale to RGB
        return np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)


def zoom_blur(x, severity_params, image_size):
    x = (np.array(x) / 255.0).astype(np.float32)
    out = np.zeros_like(x)
    for zoom_factor in severity_params:
        out += clipped_zoom(x, zoom_factor)

    x = (x + out) / (len(severity_params) + 1)
    return np.clip(x, 0, 1) * 255


def fog(x, severity_params, image_size):
    x = np.array(x) / 255.0
    max_val = x.max()
    x += (
        severity_params[0]
        * plasma_fractal(wibbledecay=severity_params[1], image_size=image_size)[
            :image_size, :image_size
        ][..., np.newaxis]
    )
    return np.clip(x * max_val / (max_val + severity_params[0]), 0, 1) * 255


def frost(x, severity_params, image_size):
    idx = np.random.randint(5)
    filename = [
        "src/data_tools/filters/frost2.png",
        "src/data_tools/filters/frost3.png",
        "src/data_tools/filters/frost1.png",
        "src/data_tools/filters/frost4.jpg",
        "src/data_tools/filters/frost5.jpg",
        "src/data_tools/filters/frost6.jpg",
    ][idx]
    # TODO: this is a bit dirty. We have a non reproducible bug here and we need to find out what's what.
    while True:
        try:
            frost = cv2.imread(filename)
            if image_size <= 32:
                frost = cv2.resize(frost, (0, 0), fx=0.2, fy=0.2)
            break
        except cv2.error:
            logger.warning(
                f"Error trying to read {filename}. Maybe it was locked by an other process?"
            )
            time.sleep(1)
            logger.info("Retrying...")

    # randomly crop and convert to rgb
    x_start, y_start = (
        np.random.randint(0, frost.shape[0] - image_size),
        np.random.randint(0, frost.shape[1] - image_size),
    )
    frost = frost[x_start : x_start + image_size, y_start : y_start + image_size][
        ..., [2, 1, 0]
    ]

    return np.clip(
        severity_params[0] * np.array(x) + severity_params[1] * frost, 0, 255
    )


def snow(x, severity_params, image_size):
    x = np.array(x, dtype=np.float32) / 255.0
    snow_layer = np.random.normal(
        size=x.shape[:2], loc=severity_params[0], scale=severity_params[1]
    )  # [:2] for monochrome

    snow_layer = clipped_zoom(snow_layer[..., np.newaxis], severity_params[2])
    snow_layer[snow_layer < severity_params[3]] = 0

    snow_layer = PILImage.fromarray(
        (np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode="L"
    )
    output = BytesIO()
    snow_layer.save(output, format="PNG")
    snow_layer = MotionImage(blob=output.getvalue())

    snow_layer.motion_blur(
        radius=severity_params[4],
        sigma=severity_params[5],
        angle=np.random.uniform(-135, -45),
    )

    snow_layer = (
        cv2.imdecode(
            np.fromstring(snow_layer.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED
        )
        / 255.0
    )
    snow_layer = snow_layer[..., np.newaxis]

    x = severity_params[6] * x + (1 - severity_params[6]) * np.maximum(
        x,
        cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).reshape(image_size, image_size, 1) * 1.5
        + 0.5,
    )
    return np.clip(x + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255


def spatter(x, severity_params, image_size):
    x = np.array(x, dtype=np.float32) / 255.0

    liquid_layer = np.random.normal(
        size=x.shape[:2], loc=severity_params[0], scale=severity_params[1]
    )

    liquid_layer = gaussian(liquid_layer, sigma=severity_params[2])
    liquid_layer[liquid_layer < severity_params[3]] = 0
    if severity_params[5] == 0:
        liquid_layer = (liquid_layer * 255).astype(np.uint8)
        dist = 255 - cv2.Canny(liquid_layer, 50, 150)
        dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
        _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
        dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
        dist = cv2.equalizeHist(dist)
        ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        dist = cv2.filter2D(dist, cv2.CV_8U, ker)
        dist = cv2.blur(dist, (3, 3)).astype(np.float32)

        m = cv2.cvtColor(liquid_layer * dist, cv2.COLOR_GRAY2BGRA)
        m /= np.max(m, axis=(0, 1))
        m *= severity_params[4]

        # water is pale turquoise
        color = np.concatenate(
            (
                175 / 255.0 * np.ones_like(m[..., :1]),
                238 / 255.0 * np.ones_like(m[..., :1]),
                238 / 255.0 * np.ones_like(m[..., :1]),
            ),
            axis=2,
        )

        color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2BGRA)

        return cv2.cvtColor(np.clip(x + m * color, 0, 1), cv2.COLOR_BGRA2BGR) * 255
    else:
        m = np.where(liquid_layer > severity_params[3], 1, 0)
        m = gaussian(m.astype(np.float32), sigma=severity_params[4])
        m[m < 0.8] = 0

        # mud brown
        color = np.concatenate(
            (
                63 / 255.0 * np.ones_like(x[..., :1]),
                42 / 255.0 * np.ones_like(x[..., :1]),
                20 / 255.0 * np.ones_like(x[..., :1]),
            ),
            axis=2,
        )

        color *= m[..., np.newaxis]
        x *= 1 - m[..., np.newaxis]

        return np.clip(x + color, 0, 1) * 255


def contrast(x, severity_params, image_size):
    x = np.array(x) / 255.0
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * severity_params + means, 0, 1) * 255


def brightness(x, severity_params, image_size):
    x = np.array(x) / 255.0
    x = sk.color.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + severity_params, 0, 1)
    x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255


def saturate(x, severity_params, image_size):
    x = np.array(x) / 255.0
    x = sk.color.rgb2hsv(x)
    x[:, :, 1] = np.clip(x[:, :, 1] * severity_params[0] + severity_params[1], 0, 1)
    x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255


def jpeg_compression(x, severity_params, image_size):
    output = BytesIO()
    x.save(output, "JPEG", quality=severity_params)
    x = PILImage.open(output)

    return x


def pixelate(x, severity_params, image_size):
    x = x.resize(
        (int(image_size * severity_params), int(image_size * severity_params)),
        PILImage.BOX,
    )
    x = x.resize((image_size, image_size), PILImage.BOX)

    return x


# mod of https://gist.github.com/erniejunior/601cdf56d2b424757de5
def elastic_transform(image, severity_params, image_size):
    c = tuple(image_size * param for param in severity_params)

    image = np.array(image, dtype=np.float32) / 255.0
    shape = image.shape
    shape_size = shape[:2]

    # random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32(
        [
            center_square + square_size,
            [center_square[0] + square_size, center_square[1] - square_size],
            center_square - square_size,
        ]
    )
    pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(
        image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101
    )

    dx = (
        gaussian(
            np.random.uniform(-1, 1, size=shape[:2]), c[1], mode="reflect", truncate=3
        )
        * c[0]
    ).astype(np.float32)
    dy = (
        gaussian(
            np.random.uniform(-1, 1, size=shape[:2]), c[1], mode="reflect", truncate=3
        )
        * c[0]
    ).astype(np.float32)
    dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = (
        np.reshape(y + dy, (-1, 1)),
        np.reshape(x + dx, (-1, 1)),
        np.reshape(z, (-1, 1)),
    )
    return (
        np.clip(
            map_coordinates(image, indices, order=1, mode="reflect").reshape(shape),
            0,
            1,
        )
        * 255
    )


# /////////////// End Distortions ///////////////

PERTURBATIONS = collections.OrderedDict()
PERTURBATIONS["Gaussian Noise"] = gaussian_noise
PERTURBATIONS["Shot Noise"] = shot_noise
PERTURBATIONS["Impulse Noise"] = impulse_noise
PERTURBATIONS["Defocus Blur"] = defocus_blur
PERTURBATIONS["Glass Blur"] = glass_blur
PERTURBATIONS["Motion Blur"] = motion_blur
PERTURBATIONS["Zoom Blur"] = zoom_blur
PERTURBATIONS["Snow"] = snow
PERTURBATIONS["Frost"] = frost
PERTURBATIONS["Fog"] = fog
PERTURBATIONS["Brightness"] = brightness
PERTURBATIONS["Contrast"] = contrast
PERTURBATIONS["Elastic"] = elastic_transform
PERTURBATIONS["Pixelate"] = pixelate
PERTURBATIONS["JPEG"] = jpeg_compression

PERTURBATIONS["Speckle Noise"] = speckle_noise
PERTURBATIONS["Gaussian Blur"] = gaussian_blur
PERTURBATIONS["Spatter"] = spatter
PERTURBATIONS["Saturate"] = saturate
