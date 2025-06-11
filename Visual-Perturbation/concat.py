import cv2
import numpy as np
import os
import random

def concat_images_horizontal(img1, img2):
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape
    
    if h1 != h2:
        img2 = cv2.resize(img2, (int(w2 * (h1 / h2)), h1), interpolation=cv2.INTER_CUBIC)
        h2, w2, _ = img2.shape
    
    half_width = max(w1, w2) // 2
    img1_resized = cv2.resize(img1, (half_width, h1), interpolation=cv2.INTER_CUBIC)
    img2_resized = cv2.resize(img2, (half_width, h2), interpolation=cv2.INTER_CUBIC)
    
    concatenated_img = np.hstack((img1_resized, img2_resized))
    
    return concatenated_img

def process_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]

    for img_file in image_files:
        img_path = os.path.join(input_dir, img_file)
        print('???')
        img1 = cv2.imread(img_path)
        if img1 is None:
            print(f"无法加载图片: {img_path}")
            continue
        
        other_images = [f for f in image_files if f != img_file]
        if not other_images:
            print(f"没有其他图片可以与 {img_file} 拼接。")
            continue
        
        other_img_file = random.choice(other_images)
        other_img_path = os.path.join(input_dir, other_img_file)
        
        img2 = cv2.imread(other_img_path)
        if img2 is None:
            print(f"无法加载图片: {other_img_path}")
            continue
        
        concatenated_img = concat_images_horizontal(img1, img2)
        
        output_path = os.path.join(output_dir, img_file)
        cv2.imwrite(output_path, concatenated_img)
        print(f"拼接图片已保存到: {output_path}")

if __name__ == "__main__":
    input_dir = "/DATA/yuting/huggingface/thinklite11k/images"
    output_dir = "/DATA/yuting/huggingface/thinklite11k-concat/images"

    process_images(input_dir, output_dir)