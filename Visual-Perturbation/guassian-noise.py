import cv2
import numpy as np
import os

def add_gaussian_noise(image, mean=0, std=25):
    noise = np.random.normal(loc=mean, scale=std, size=image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image

def process_images_with_noise(input_dir, output_dir, mean=0, std=25):
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

        noisy_img = add_gaussian_noise(img, mean=mean, std=std)

        output_path = os.path.join(output_dir, img_file)
        cv2.imwrite(output_path, noisy_img)
        print(f"添加噪声的图片已保存到: {output_path}")

if __name__ == "__main__":
    input_dir = "/DATA/datasets/geoqa-r1v-8k/image"
    output_dir = "/DATA/yuting/datasets/geoqa-r1v-noise/image"

    mean = 0
    std = 300
    process_images_with_noise(input_dir, output_dir, mean=mean, std=std)