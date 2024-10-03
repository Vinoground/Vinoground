import torch
import pandas as pd
from tqdm import tqdm
from languagebind import LanguageBindVideo, LanguageBindVideoTokenizer, LanguageBindVideoProcessor

import argparse
import os

# Create an ArgumentParser object
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument('--data', type=str, default="./Vinoground", help='Path to Vinoground dataset (from Huggingface)')
parser.add_argument('--ckpt', type=str, default="./checkpoints/langbind", help='Path to model checkpoints')

# Parse arguments
args = parser.parse_args()

data_path = args.data
ckpt_path = args.ckpt

vino = pd.read_csv(os.path.join(data_path, "vinoground.csv"))

pretrained_ckpt = ckpt_path  # also 'LanguageBind/LanguageBind_Video'
model = LanguageBindVideo.from_pretrained(pretrained_ckpt, cache_dir='./cache_dir')
tokenizer = LanguageBindVideoTokenizer.from_pretrained("LanguageBind/LanguageBind_Video_FT", cache_dir='./cache_dir')
video_process = LanguageBindVideoProcessor(model.config, tokenizer)

model.eval()

text_correct = 0
video_correct = 0
group_correct = 0

videos = []
texts = []
for i in range(500):
    videos.append(os.path.join(data_path, f"vinoground_videos/{i}_pos.mp4"))
    videos.append(os.path.join(data_path, f"vinoground_videos/{i}_neg.mp4"))
    texts.append(vino["pos_cap"][i])
    texts.append(vino["neg_cap"][i])
data = video_process(videos, texts, return_tensors='pt')
with torch.no_grad():
    out = model(**data)
results = out.text_embeds @ out.image_embeds.T

for i in tqdm(range(0, 1000, 2)):
    text_correct += results[i][i] > results[i+1][i] and results[i+1][i+1] > results[i][i+1]
    video_correct += results[i][i] > results[i][i+1] and results[i+1][i+1] > results[i+1][i]
    group_correct += results[i][i] > results[i+1][i] and results[i+1][i+1] > results[i][i+1] and results[i][i] > results[i][i+1] and results[i+1][i+1] > results[i+1][i]
    
print(text_correct / 500, video_correct / 500, group_correct / 500)