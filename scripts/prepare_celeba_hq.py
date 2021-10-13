# donwload dataset from https://drive.google.com/open?id=1badu11NqxGf6qM3PTTooQDJvQbejgbTv
# (see https://github.com/switchablenorms/CelebAMask-HQ)
# download .txt files from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
import pandas as pd
import numpy as np
import os

CELEBA_PATH = "/scratch/s193223/celeba/"
CELEBAHQ_PATH = "/scratch/s193223/celebahq2/CelebAMask-HQ/"

df = pd.read_csv(
    os.path.join(CELEBAHQ_PATH, "CelebA-HQ-to-CelebA-mapping.txt"), sep="\s+"
)

# get original splits
split = pd.read_csv(
    os.path.join(CELEBA_PATH, "list_eval_partition.txt"), sep="\s+", header=None
)
split.columns = ["orig_file", "split"]
df = pd.merge(df, split, on="orig_file", how="left")

# add 2k additional validation split
np.random.seed(0)
new_split_ids = np.random.choice(df[df.split == 0].idx, 3000, replace=False)
df.loc[df.idx.isin(new_split_ids), "split"] = 3
print(df.split.value_counts())

# get attributes
attr = pd.read_csv(
    os.path.join(CELEBA_PATH, "list_attr_celeba.txt"), sep="\s+", skiprows=1
)
df = pd.merge(df, attr.reset_index(), left_on="orig_file", right_on="index", how="left")

df["file_name"] = df.idx.apply(lambda x: str(x) + ".jpg")

df.to_csv(os.path.join(CELEBAHQ_PATH, "metadata.csv"), index=False)
