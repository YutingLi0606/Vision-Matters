import cv2
import numpy as np
import os
import random

def apply_gaussian_blur(image, max_kernel_size=7):
    kernel_size = random.choice([i for i in range(3, max_kernel_size + 1) if i % 2 == 1])
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=0)
    
    return blurred_image

def process_images_with_blur(input_dir, output_dir, max_kernel_size=7):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    supported_extensions = ('.png', '.jpg', '.jpeg')
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(supported_extensions)]

    for img_file in image_files:
        img_path = os.path.join(input_dir, img_file)

        img = cv2.imread(img_path)
        if img is None:
            print(f"无法加载图片: {img_path}")
            continue

        blurred_img = apply_gaussian_blur(img, max_kernel_size=max_kernel_size)

        output_path = os.path.join(output_dir, img_file)
        cv2.imwrite(output_path, blurred_img)

if __name__ == "__main__":
    input_dir = "/DATA/datasets/geoqa-r1v-8k/image"
    output_dir = "/DATA/yuting/datasets/geoqa-r1v-blur/image"

    max_kernel_size = 50

    process_images_with_blur(input_dir, output_dir, max_kernel_size=max_kernel_size)