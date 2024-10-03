from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import pandas as pd
import os

import argparse

# Create an ArgumentParser object
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument('--data', type=str, default="./Vinoground", help='Path to Vinoground dataset (from Huggingface)')

# Parse arguments
args = parser.parse_args()

data_path = args.data

vino = pd.read_csv(os.path.join(data_path, "vinoground.csv"))


device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Instantiate model
model = imagebind_model.imagebind_huge(True)
model.eval()
model.to(device)

text_correct = 0
video_correct = 0
group_correct = 0

from tqdm import tqdm
for i in tqdm(range(500)):
    videos = []
    texts = []
    videos.append(os.path.join(data_path, f"vinoground_videos/{i}_pos.mp4"))
    videos.append(os.path.join(data_path, f"vinoground_videos/{i}_neg.mp4"))
    texts.append(vino["pos_cap"][i])
    texts.append(vino["neg_cap"][i])
    

    inputs = {
        ModalityType.TEXT: data.load_and_transform_text(texts, device),
        ModalityType.VISION: data.load_and_transform_video_data(videos, device),
    }

    with torch.no_grad():
        embeddings = model(inputs)

    results = embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T
    # print(results)

    video_correct += results[0][0] > results[1][0] and results[1][1] > results[0][1]
    text_correct += results[0][0] > results[0][1] and results[1][1] > results[1][0]
    group_correct += results[0][0] > results[1][0] and results[1][1] > results[0][1] and results[0][0] > results[0][1] and results[1][1] > results[1][0]
        
print(text_correct / 500, video_correct / 500, group_correct / 500)