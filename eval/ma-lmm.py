import os
import torch
import numpy as np
from lavis.models import load_model_and_preprocess
import json
from tqdm import tqdm

import decord
from decord import VideoReader
from decord import cpu, gpu
decord.bridge.set_bridge('torch')
import argparse

# Create an ArgumentParser object
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument('--data', type=str, default="./Vinoground", help='Path to Vinoground dataset (from Huggingface)')
parser.add_argument('--ckpt', type=str, default="./checkpoints/ma-lmm", help='Path to model checkpoints')
parser.add_argument("--output", type=str, default="./outputs", help="Output directory of score files")
parser.add_argument("--nframes", type=int, default=32, help="Number of frames to sample.")

# Parse arguments
args = parser.parse_args()

data_path = args.data
ckpt_path = args.ckpt
output_dir = args.output
nframes = args.nframes
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def load_video(vr, start_time, end_time, fps, num_frames=20):
    start_index = int(round(start_time * fps))
    end_index = int(round(end_time * fps))
    select_frame_index = np.rint(np.linspace(start_index, end_index-1, num_frames)).astype(int).tolist()
    frames = vr.get_batch(select_frame_index).permute(3, 0, 1, 2).to(torch.float32)
    return frames

# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


model, vis_processors, _ = load_model_and_preprocess(
    name=ckpt_path, model_type="vicuna7b", is_eval=True, device=device, memory_bank_length=10, num_frames=nframes,
)

os.mkdir(os.path.join(output_dir, f"malmm-frame{nframes}"))
video_ans_file = open(os.path.join(output_dir, f"malmm-frame{nframes}", f"videoscore-response.jsonl"), 'w')
text_ans_file = open(os.path.join(output_dir, f"malmm-frame{nframes}", f"textscore-frame{nframes}-response.jsonl"), 'w')

with open(os.path.join(data_path, "vinoground_textscore.json"), 'r') as f:
    questions = json.load(f)

for question in tqdm(questions):
    try:

        file_path = os.path.join(data_path, question["video_name"])
        vr = VideoReader(file_path, ctx=cpu(0))
        total_frames = len(vr)
        fps = vr.get_avg_fps()
        duration = total_frames / fps

        video = load_video(vr, start_time=0, end_time=duration, fps=fps, num_frames=nframes)

        video = vis_processors["eval"](video).to(device).unsqueeze(0)
        output = model.generate({"image": video, "prompt": "Question: " + question["question"] + "Please only output 1 English character. Answer: "})
        
        text_ans_file.write(json.dumps(dict(idx=question["idx"], response=output)) + '\n')
        text_ans_file.flush()
    except Exception as e:
        print(e)
        continue

text_ans_file.close()


with open(os.path.join(data_path, "vinoground_videoscore.json"), 'r') as f:
    questions = json.load(f)

for question in tqdm(questions):
    try:

        file_path = os.path.join(data_path, question["video_name"])
        vr = VideoReader(file_path, ctx=cpu(0))
        total_frames = len(vr)
        fps = vr.get_avg_fps()
        duration = total_frames / fps

        video = load_video(vr, start_time=0, end_time=duration, fps=fps, num_frames=nframes)

        video = vis_processors["eval"](video).to(device).unsqueeze(0)
        output = model.generate({"image": video, "prompt": "Question: " + question["question"] + "Please only output 1 English character. Answer: "})
        
        video_ans_file.write(json.dumps(dict(idx=question["idx"], response=output)) + '\n')
        video_ans_file.flush()
    except Exception as e:
        print(e)
        continue

video_ans_file.close()