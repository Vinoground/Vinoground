import sys
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

def load_video(vr, start_time, end_time, fps, num_frames=20):
    start_index = int(round(start_time * fps))
    end_index = int(round(end_time * fps))
    select_frame_index = np.rint(np.linspace(start_index, end_index-1, num_frames)).astype(int).tolist()
    frames = vr.get_batch(select_frame_index).permute(3, 0, 1, 2).to(torch.float32)
    return frames

# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

for nframes in [32, 16, 8, 4, 2]:
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_vicuna_instruct_malmm", model_type="vicuna7b", is_eval=True, device=device, memory_bank_length=10, num_frames=nframes,
    )

    video_ans_file = open(f"ma-lmm-vicuna-7b-videoscore-frame{nframes}-response.jsonl", 'w')
    text_ans_file = open(f"ma-lmm-vicuna-7b-textscore-frame{nframes}-response.jsonl", 'w')
    
    with open("vinoground_textscore.json", 'r') as f:
        questions = json.load(f)

    for question in tqdm(questions):
        try:

            file_path = "../" + question["video_name"]
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


    with open("vinoground_videoscore.json", 'r') as f:
        questions = json.load(f)

    for question in tqdm(questions):
        try:

            file_path = "../" + question["video_name"]
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