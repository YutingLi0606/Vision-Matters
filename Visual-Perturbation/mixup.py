import os
import random
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms



image_dir = "/DATA/datasets/geoqa-r1v-8k/image"
output_dir = "/DATA/yuting/datasets/geoqa-r1v-mixup/image"
os.makedirs(output_dir, exist_ok=True)

target_size = (256, 256)

def load_and_pad_image(image_path, target_size):
    image = Image.open(image_path).convert('RGB')
    width, height = image.size
    target_width, target_height = target_size
    ratio = min(target_width / width, target_height / height)
    new_size = (int(width * ratio), int(height * ratio))
    
    image = image.resize(new_size, Image.Resampling.LANCZOS) 
    
    padded_image = Image.new('RGB', target_size, (255, 255, 255))  
    padded_image.paste(image, ((target_width - new_size[0]) // 2, (target_height - new_size[1]) // 2))
    
    transform = transforms.ToTensor()
    return transform(padded_image)

def mixup_data(x1, x2, lambda_min=0.7, lambda_max=0.9):

    lam = np.random.uniform(lambda_min, lambda_max) 
    mixed_x = lam * x1 + (1 - lam) * x2 
    return mixed_x, lam

def save_image(tensor, path):
    image = transforms.ToPILImage()(tensor)
    image.save(path)

image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.png')]
num_images = len(image_paths)

for i, main_image_path in enumerate(image_paths):

    main_image = load_and_pad_image(main_image_path, target_size)
    sub_image_index = random.choice([j for j in range(num_images) if j != i])
    sub_image_path = image_paths[sub_image_index]
    sub_image = load_and_pad_image(sub_image_path, target_size)
    #mixed_x, lam = mixup_data(main_image, sub_image, lambda_min=0.45, lambda_max=0.55)
    mixed_x, lam = mixup_data(main_image, sub_image)
    main_image_name = os.path.basename(main_image_path)
    output_path = os.path.join(output_dir, main_image_name)
    save_image(mixed_x, output_path)
    
    print(f"Saved mixup image: {main_image_name}, lambda={lam:.2f}")

print("Data augmentation completed!")