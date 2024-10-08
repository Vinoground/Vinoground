from PIL import Image
from transformers import AutoModel, AutoTokenizer
from decord import VideoReader, cpu    # pip install decord
from tqdm import tqdm
import json

import argparse

# Create an ArgumentParser object
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument('--data', type=str, default="./Vinoground", help='Path to Vinoground dataset (from Huggingface)')
parser.add_argument('--ckpt', type=str, default="./checkpoints/MiniCPM-V-2_6", help='Path to model checkpoints')
parser.add_argument("--output", type=str, default="./outputs/", help="Output directory of score files")
parser.add_argument("--nframes", type=int, default=32, help="Number of frames to sample.")
# Parse arguments
args = parser.parse_args()

data_path = args.data
ckpt_path = args.ckpt
output_dir = args.output
nframes = args.nframes
import os
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model = AutoModel.from_pretrained(ckpt_path, trust_remote_code=True,
    attn_implementation='flash_attention_2', torch_dtype="auto") # sdpa or flash_attention_2, no eager
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True)

MAX_NUM_FRAMES=64 # if cuda OOM set a smaller number

def encode_video(video_path):
    global nframes
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    # frame_idx = [i for i in range(0, len(vr), sample_fps)]
    frame_idx = [i for i in range(0, len(vr), len(vr) // nframes)]
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    # print('num frames:', len(frames))
    return frames


# Set decode params for video
params = {}
params["use_image_id"] = False
params["max_slice_nums"] = 2 # use 1 if cuda OOM and video resolution > 448*448


os.mkdir(os.path.join(output_dir, f"minicpm-frame{nframes}"))
video_ans_file = open(os.path.join(output_dir, f"minicpm-frame{nframes}", f"videoscore-response.jsonl"), 'w')
text_ans_file = open(os.path.join(output_dir, f"minicpm-frame{nframes}", f"textscore-frame{nframes}-response.jsonl"), 'w')

with open(os.path.join(data_path, "vinoground_textscore.json"), 'r') as f:
    questions = json.load(f)

for question in tqdm(questions):
    try:

        frames = encode_video(os.path.join(data_path, question["video_name"]))
        msgs = [
            {
                'role': 'user', 
                'content': frames + [question["question"] + "Please only output one English character."]
            }, 
        ]

        answer = model.chat(
            image=None,
            msgs=msgs,
            tokenizer=tokenizer,
            **params
        )
        
        text_ans_file.write(json.dumps(dict(idx=question["idx"], response=answer)) + '\n')
        text_ans_file.flush()

    except Exception as e:
        print(e)
        continue


text_ans_file.close()


with open(os.path.join(data_path, "vinoground_videoscore.json"), 'r') as f:
    questions = json.load(f)

for question in tqdm(questions):
    try:
        frames = encode_video(os.path.join(data_path, question["video_name"]))
        msgs = [
            {
                'role': 'user', 
                'content': frames + [question["question"] + "Please only output one English character."]
            }, 
        ]

        answer = model.chat(
            image=None,
            msgs=msgs,
            tokenizer=tokenizer,
            **params
        )
        
        video_ans_file.write(json.dumps(dict(idx=question["idx"], response=answer)) + '\n')
        video_ans_file.flush()

    except Exception as e:
        print(e)
        continue


video_ans_file.close()