import cv2

def open_img(img_path):
    cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

def preprocess(img):
    pass