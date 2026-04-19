import os

import cv2
import numpy as np
from skimage.feature import local_binary_pattern


def load_image(path, grayscale=False):
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(path, flag)
    if image is None:
        raise FileNotFoundError(f"Não foi possível abrir a imagem: {path}")
    return image


def load_grayscale(path):
    return load_image(path, grayscale=True)


def save_image(path, image):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, image)


def convert_to_gray(image):
    if image is None:
        raise ValueError("Imagem inválida")
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def normalize_output(image):
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        image = image.astype(np.uint8)
    return image


def apply_gaussian_blur(image, ksize=5, sigma=1.0):
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def sobel_edge(image, ksize=3):
    gray = convert_to_gray(image)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    return normalize_output(magnitude)


def prewitt_edge(image):
    gray = convert_to_gray(image).astype(np.float32)
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
    kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
    gx = cv2.filter2D(gray, -1, kernelx)
    gy = cv2.filter2D(gray, -1, kernely)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    return normalize_output(magnitude)


def laplacian_edge(image, ksize=3):
    gray = convert_to_gray(image)
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
    return normalize_output(np.abs(lap))


def canny_edge(image, low_threshold=50, high_threshold=150):
    gray = convert_to_gray(image)
    return cv2.Canny(gray, low_threshold, high_threshold)


def threshold_binary(image, threshold=128):
    gray = convert_to_gray(image)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary


def adaptive_threshold(image, block_size=11, c=2):
    gray = convert_to_gray(image)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c)


def _get_structuring_element(size):
    return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))


def dilate(image, kernel_size=3, iterations=1):
    kernel = _get_structuring_element(kernel_size)
    return cv2.dilate(image, kernel, iterations=iterations)


def erode(image, kernel_size=3, iterations=1):
    kernel = _get_structuring_element(kernel_size)
    return cv2.erode(image, kernel, iterations=iterations)


def opening(image, kernel_size=3, iterations=1):
    kernel = _get_structuring_element(kernel_size)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)


def closing(image, kernel_size=3, iterations=1):
    kernel = _get_structuring_element(kernel_size)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def hit_or_miss(binary_image, kernel):
    if binary_image.dtype != np.uint8:
        raise ValueError("A imagem deve ser binária em uint8")
    unique_values = np.unique(binary_image)
    if not set(unique_values).issubset({0, 255}):
        raise ValueError("A imagem deve conter apenas valores 0 e 255")

    kernel = np.array(kernel, dtype=np.int8)
    if kernel.ndim != 2:
        raise ValueError("O kernel deve ser uma matriz 2D")

    foreground = (kernel == 1).astype(np.uint8)
    background = (kernel == -1).astype(np.uint8)

    if foreground.sum() == 0 and background.sum() == 0:
        raise ValueError("O kernel deve conter pelo menos um 1 ou -1")

    binary = (binary_image == 255).astype(np.uint8)
    complement = 1 - binary

    if foreground.sum() > 0:
        fg_eroded = cv2.erode(binary, foreground, iterations=1)
    else:
        fg_eroded = np.ones_like(binary)

    if background.sum() > 0:
        bg_eroded = cv2.erode(complement, background, iterations=1)
    else:
        bg_eroded = np.ones_like(binary)

    hit = cv2.bitwise_and(fg_eroded, bg_eroded)
    return (hit * 255).astype(np.uint8)


def get_default_hit_or_miss_kernel():
    return [
        [0, 1, 0],
        [-1, 1, -1],
        [0, 1, 0],
    ]


def color_histogram_features(image, bins=32):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    features = {}
    ranges = [(0, 180), (0, 256), (0, 256)]
    channels = ["h", "s", "v"]
    for idx, channel in enumerate(channels):
        hist = cv2.calcHist([hsv], [idx], None, [bins], ranges[idx])
        hist = cv2.normalize(hist, hist).flatten()
        for i, value in enumerate(hist):
            features[f"{channel}_hist_{i}"] = float(value)
    return features


def texture_features(image, bins=16, radius=1, points=8):
    gray = convert_to_gray(image)
    lbp = local_binary_pattern(gray, points, radius, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, bins))
    hist = hist.astype(np.float32)
    hist /= hist.sum() + 1e-8
    return {f"lbp_{i}": float(hist[i]) for i in range(len(hist))}


def edge_statistics(image):
    edge = sobel_edge(image)
    edge_pixels = np.count_nonzero(edge > 0)
    mean_strength = float(edge[edge > 0].mean()) if edge_pixels else 0.0
    return {
        "edge_pixels": int(edge_pixels),
        "edge_mean_strength": mean_strength,
    }


def shape_features(image, threshold=128):
    gray = convert_to_gray(image)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            "shape_area": 0,
            "shape_perimeter": 0,
            "shape_aspect_ratio": 0,
            "shape_extent": 0,
            "shape_solidity": 0,
        }
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    x, y, w, h = cv2.boundingRect(contour)
    rect_area = w * h if w * h else 1
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull) or 1
    return {
        "shape_area": float(area),
        "shape_perimeter": float(perimeter),
        "shape_aspect_ratio": float(w / h) if h else 0.0,
        "shape_extent": float(area / rect_area),
        "shape_solidity": float(area / hull_area),
    }


def extract_image_features(image, mask=None):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    features = {}
    features.update(color_histogram_features(image, bins=16))
    features.update(texture_features(image, bins=16))
    features.update(edge_statistics(image))
    features.update(shape_features(image))
    return features


def extract_edge_features(edge_image):
    edge_pixels = np.count_nonzero(edge_image > 0)
    total_pixels = edge_image.size
    density = edge_pixels / total_pixels
    return {
        "edge_pixel_count": int(edge_pixels),
        "edge_density": float(density),
    }


def extract_features_from_path(path):
    image = load_image(path)
    return extract_image_features(image)
