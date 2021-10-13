import os
from PIL import Image
from tqdm import tqdm

SOURCE_PATH = "/scratch/s193223/celebahq2/CelebAMask-HQ/CelebA-HQ-img"
DESTINATION = "/scratch/s193223/celebahq2/CelebAMask-HQ/img256"
TARGET_RES = 256

if not os.path.exists(DESTINATION):
    os.mkdir(DESTINATION)

for f in tqdm(os.listdir(SOURCE_PATH)):
    if f.endswith('.jpg') or f.endswith('.png'):
        img = Image.open(os.path.join(SOURCE_PATH, f))
        img = img.resize((TARGET_RES, TARGET_RES))
        img.save(os.path.join(DESTINATION, f))