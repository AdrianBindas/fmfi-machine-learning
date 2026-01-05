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
    Compute average symmetry score for all angles 0 - 359 degrees.
    """
    img = load_image(img_path, grayscale=True)
    scores = {}

    for angle in range(360):
        reflected = reflect_about_angle(img, angle)
        diff = np.abs(img - reflected)
        score = diff.sum() / diff.size
        scores[angle] = score

    return sum(scores.values()) / 360


def contour_area(img_path):
    """
    Calculate area of the largest contour.
    """
    img = load_image(img_path, grayscale=True)
    contours, hierarchy = cv2.findContours(image=img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return 0.0
    
    largest_contour = max(contours, key=cv2.contourArea)
    return float(cv2.contourArea(largest_contour))


def contour_perimeter(img_path):
    """
    Calculate perimeter of the largest contour.
    """
    img = load_image(img_path, grayscale=True)
    contours, hierarchy = cv2.findContours(image=img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return 0.0
    
    largest_contour = max(contours, key=cv2.contourArea)
    return float(cv2.arcLength(largest_contour, True))


def grayscale_variance(img_path, mask_path):
    """
    Convert image to grayscale and calculate variance of an image within the mask.
    """
    img = load_image(img_path, grayscale=True)
    mask = load_image(mask_path, grayscale=True)

    mask_bool = mask.astype(bool)
    if not np.any(mask_bool):
        return 0.0
    
    masked_values = img[mask_bool].astype(np.float32)
    var = np.var(masked_values)
    
    norm_var = var / (255**2)
    
    return float(norm_var)


def color_variance(img_path, mask_path):
    """
    Calculate variance of an image, average variance of R,G,B channels.
    """
    img = load_image(img_path)
    mask = load_image(mask_path, grayscale=True)

    mask_bool = mask.astype(bool)
    if not np.any(mask_bool):
        return 0.0

    variances = []
    for channel in range(3):
        channel_data = img[:, :, channel]
        masked_values = channel_data[mask_bool].astype(np.float32)
        var = np.var(masked_values)
        norm_var = var / (255**2)
        variances.append(norm_var)
    
    return float(np.mean(variances))


def color_histogram(img_path, mask_path, bins=32):
    """
    Calculate color histogram of an image within the mask.
    """
    img = load_image(img_path)
    mask = load_image(mask_path, grayscale=True)
    
    histograms = []
    for channel in range(3):
        hist = cv2.calcHist(
            [img], 
            [channel], 
            mask, 
            [bins], 
            [0, 256]
        )
        hist = hist.flatten() / (hist.sum() + 1e-7)
        histograms.append(hist)
    
    return np.concatenate(histograms)


def contour_approximate(img_path, epsilon_factor=0.05):
    """
    Approximate image contour using cv2.approxPolyDP.
    """
    img = load_image(img_path, grayscale=True)
    contours, hierarchy = cv2.findContours(
        image=img, 
        mode=cv2.RETR_EXTERNAL, 
        method=cv2.CHAIN_APPROX_NONE
    )
    
    if not contours:
        return 0
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    perimeter = cv2.arcLength(largest_contour, True)
    
    epsilon = epsilon_factor * perimeter
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    return int(len(approx))


def solidity(img_path):
    """
    Solidity = contour area / convex hull area
    """
    img = load_image(img_path, grayscale=True)
    contours, hierarchy = cv2.findContours(
        image=img, 
        mode=cv2.RETR_EXTERNAL, 
        method=cv2.CHAIN_APPROX_NONE
    )
    
    if not contours:
        return 0.0
    
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    
    hull = cv2.convexHull(largest_contour)
    hull_area = cv2.contourArea(hull)
    
    if hull_area == 0:
        return 0.0
    
    solidity = area / hull_area
    return float(solidity)
