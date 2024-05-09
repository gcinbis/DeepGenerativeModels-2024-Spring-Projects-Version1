import os
from PIL import Image
from tqdm import tqdm

def downscale_by_k_bicubic(image, k=4):
    width, height = image.size
    return image.resize((width // k, height // k), Image.BICUBIC)

def main(path, save_path, k=4):
    for img_name in tqdm(os.listdir(path)):
        img_path = os.path.join(path, img_name)
        img = Image.open(img_path)
        img_resized = downscale_by_k_bicubic(img, k)
        img_resized.save(os.path.join(save_path, f"{os.path.splitext(img_name)[0]}_downscaled.png"))

PATH = "data/evaluation/hr/Set5"
SAVE_PATH = "data/evaluation/lr/Set5"

if __name__ == "__main__":
    main(PATH, SAVE_PATH)