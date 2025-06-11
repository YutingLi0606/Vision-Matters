import cv2
import numpy as np
import os
import random

def rotate_image(image, angle, fill_color=(255, 255, 255)):

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]

    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=fill_color)

    return rotated_image

def process_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    
    for img_file in image_files:
        img_path = os.path.join(input_dir, img_file)
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法加载图片: {img_path}")
            continue
        
        angle = random.uniform(-180, 180)
        
        rotated_img = rotate_image(img, angle, fill_color=(255, 255, 255))

        output_path = os.path.join(output_dir, img_file)
        cv2.imwrite(output_path, rotated_img)
        print(f"旋转图片已保存到: {output_path}，旋转角度: {angle:.2f}°")

if __name__ == "__main__":
    input_dir = "/DATA/datasets/geoqa-r1v-8k/image"
    output_dir = "/DATA/yuting/datasets/geoqa-r1v-rotate/image"
    process_images(input_dir, output_dir)