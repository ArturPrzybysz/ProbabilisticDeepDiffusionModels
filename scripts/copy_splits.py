import os
from shutil import copyfile
from tqdm import tqdm
import pandas as pd

CELEBAHQ_PATH = "/scratch/s193223/celebahq2/CelebAMask-HQ/"
SOURCE = "/scratch/s193223/celebahq2/CelebAMask-HQ/img256"
PATH_TRAIN = "/scratch/s193223/celebahq2/CelebAMask-HQ/img256train"
PATH_VAL = "/scratch/s193223/celebahq2/CelebAMask-HQ/img256val"

if not os.path.exists(PATH_TRAIN):
    os.mkdir(PATH_TRAIN)
if not os.path.exists(PATH_VAL):
    os.mkdir(PATH_VAL)

df = pd.read_csv(os.path.join(CELEBAHQ_PATH, "metadata.csv"))

print(df.split.value_counts())
for _, row in tqdm(df.iterrows()):
    f = row.file_name
    if row.split == 0:
        copyfile(os.path.join(SOURCE, f), os.path.join(PATH_TRAIN, f))
    else:
        copyfile(os.path.join(SOURCE, f), os.path.join(PATH_VAL, f))

print("train:", len(os.listdir(PATH_TRAIN)))
print("val:", len(os.listdir(PATH_VAL)))
