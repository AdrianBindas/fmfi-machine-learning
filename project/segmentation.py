import cv2
from utils import load_image
import numpy as np

def apply_clahe(img_path, clip_limit=10.0, tile_grid_size=(20, 20)):
    img = load_image(img_path)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    v_clahe = clahe.apply(v)
    hsv_clahe = cv2.merge((h, s, v_clahe))
    bgr_clahe = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)
    
    return bgr_clahe


def apply_median_blur(img_path, kernel_size=11):
    img = load_image(img_path)
    return cv2.medianBlur(img, kernel_size)


def apply_otsu_threshold(img_path):
    img = load_image(img_path, grayscale=True)
    gray_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    return cv2.bitwise_not(gray_otsu)


def apply_morph(img_path):
    img = load_image(img_path)
    kernel = img.shape[0] // 100, img.shape[1] // 100
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)


def find_max_contour(img_path):
    img = load_image(img_path, grayscale=True)
    contours, hierarchy = cv2.findContours(image=img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    largest_contour = max(contours, key=cv2.contourArea)
    return cv2.drawContours(image=np.zeros_like(img), contours=[largest_contour], contourIdx=-1, color=255, thickness=cv2.FILLED)
