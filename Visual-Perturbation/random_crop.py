import cv2
import numpy as np
import os
import random

def random_proportional_crop(image, min_scale=0.6, max_scale=0.9):

    h, w = image.shape[:2]

    crop_width = int(random.uniform(min_scale, max_scale) * w)
    crop_height = int(random.uniform(min_scale, max_scale) * h)

    crop_width = min(crop_width, w)
    crop_height = min(crop_height, h)

    x_start = random.randint(0, w - crop_width)
    y_start = random.randint(0, h - crop_height)

    cropped_image = image[y_start:y_start+crop_height, x_start:x_start+crop_width]
    return cropped_image

def process_images_with_random_proportional_crop(input_dir, output_dir, min_scale=0.6, max_scale=0.9):
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

        cropped_img = random_proportional_crop(img, min_scale=min_scale, max_scale=max_scale)

        output_path = os.path.join(output_dir, img_file)
        cv2.imwrite(output_path, cropped_img)
        print(f"已保存裁剪后的图片到: {output_path}，裁剪尺寸: {cropped_img.shape[1]}x{cropped_img.shape[0]}")

if __name__ == "__main__":

    input_dir = "/DATA/datasets/geoqa-r1v-8k/image"
    output_dir = "/DATA/yuting/datasets/geoqa-r1v-crop/image"

    min_scale = 0.6
    max_scale = 0.9

    process_images_with_random_proportional_crop(input_dir, output_dir, min_scale=min_scale, max_scale=max_scale)

