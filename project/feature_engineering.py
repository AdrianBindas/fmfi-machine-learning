import numpy as np
from skimage.transform import rotate
import cv2
from utils import load_image


def reflect_about_angle(img, angle_deg):
    """
    Reflect image about an axis through the center at angle_deg
    """
    rotated = rotate(
        img,
        -angle_deg,
        resize=False,
        order=0,
        preserve_range=True
    )

    reflected = np.fliplr(rotated)

    restored = rotate(
        reflected,
        angle_deg,
        resize=False,
        order=0,
        preserve_range=True
    )

    return (restored > 124).astype(bool)


def naive_symmetry(img_path):
    """
    Compute symmetry scores for all angles 0 - 359 degrees
    """
    img = load_image(img_path, grayscale=True)
    scores = {}

    for angle in range(360):
        reflected = reflect_about_angle(img, angle)
        diff = np.abs(img - reflected)
        score = diff.sum() / diff.size
        scores[angle] = score

    return scores.sum() / 360


def contour_area(img_path):
    """
    Calculate area of the largest contour.
    """
    img = load_image(img_path, grayscale=True)
    contours, hierarchy = cv2.findContours(image=img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    largest_contour = max(contours, key=cv2.contourArea)
    
    return cv2.contourArea(largest_contour)


def contour_perimeter(img_path):
    """
    Calculate perimeter of the largest contour.
    """
    img = load_image(img_path, grayscale=True)
    contours, hierarchy = cv2.findContours(image=img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    largest_contour = max(contours, key=cv2.contourArea)
    
    return cv2.arcLength(largest_contour,True)


def grayscale_variance(img_path, mask_path):
    """
    Convert image to grayscale and calculate variance of an image
    """
    img = load_image(img_path, grayscale=True)
    mask = load_image(mask_path, grayscale=True)
    masked = np.where(mask.astype(bool), img, 0)
    maskedf = masked.astype(np.float32)
    meanf = np.mean(maskedf)
    maskedf2 = maskedf**2
    meanf2 = np.mean(maskedf2)
    var = meanf2 - meanf**2
    norm_var = var / (255**2)
    return norm_var


def color_variance(img_path, mask_path):
    """
    Calculate variance of an image, average variance of R,G,B channels
    """
    ...


def color_histogram(img_path, mask_path):
    """
    Calculate color histogram of an image
    """
    ...


def contour_approximate(img_path):
    """
    Approximate image contour using cv2.approxPolyDP
    """
    ...