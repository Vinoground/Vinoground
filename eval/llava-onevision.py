from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

import numpy as np
import copy
import warnings
from decord import VideoReader, cpu
import json
from tqdm import tqdm
import argparse

# Create an ArgumentParser object
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument('--data', type=str, default="./Vinoground", help='Path to Vinoground dataset (from Huggingface)')
parser.add_argument('--ckpt', type=str, default="./checkpoints/llava-one-vision-qwen2-72b-ov-sft", help='Path to model checkpoints')
parser.add_argument("--output", type=str, default="./outputs", help="Output directory of score files")
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


warnings.filterwarnings("ignore")
# Load the OneVision model
pretrained = ckpt_path
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, attn_implementation="sdpa")

model.eval()


# Function to extract frames from video
def load_video(video_path, max_frames_num):
    if type(video_path) == str:
        vr = VideoReader(video_path, ctx=cpu(0))
    else:
        vr = VideoReader(video_path[0], ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames  # (frames, height, width, channels)


os.mkdir(os.path.join(output_dir, f"llavaonevision-frame{nframes}"))
video_ans_file = open(os.path.join(output_dir, f"llavaonevision-frame{nframes}", f"videoscore-response.jsonl"), 'w')
text_ans_file = open(os.path.join(output_dir, f"llavaonevision-frame{nframes}", f"textscore-frame{nframes}-response.jsonl"), 'w')

with open(os.path.join(data_path, "vinoground_textscore.json"), 'r') as f:
    questions = json.load(f)

for question in tqdm(questions):
    try:
        # Load and process video
        video_path = os.path.join(data_path, question["video_name"])
        video_frames = load_video(video_path, nframes)
        # print(video_frames.shape) # (16, 1024, 576, 3)
        image_tensors = []
        frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
        image_tensors.append(frames)

        # Prepare conversation input
        conv_template = "qwen_2"
        prompt = f"{DEFAULT_IMAGE_TOKEN}\n" + question["question"] + "\nPlease only output one English character."

        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        image_sizes = [frame.size for frame in video_frames]

        # Generate response
        cont = model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
            modalities=["video"],
        )
        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)

        text_ans_file.write(json.dumps(dict(idx=question["idx"], response=text_outputs[0])) + '\n')
        text_ans_file.flush()
    except Exception as e:
        print(e)
        continue


text_ans_file.close()


with open(os.path.join(data_path, "vinoground_videoscore.json"), 'r') as f:
    questions = json.load(f)

for question in tqdm(questions):
    try:
        
        # Load and process video
        video_path = os.path.join(data_path, question["video_name"])
        video_frames = load_video(video_path, nframes)
        # print(video_frames.shape) # (16, 1024, 576, 3)
        image_tensors = []
        frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
        image_tensors.append(frames)

        # Prepare conversation input
        conv_template = "qwen_2"
        prompt = f"{DEFAULT_IMAGE_TOKEN}\n" + question["question"] + "\nPlease only output one English character."

        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        image_sizes = [frame.size for frame in video_frames]

        # Generate response
        cont = model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
            modalities=["video"],
        )
        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)

        video_ans_file.write(json.dumps(dict(idx=question["idx"], response=text_outputs[0])) + '\n')
        video_ans_file.flush()
    except Exception as e:
        print(e)
        continue


video_ans_file.close()