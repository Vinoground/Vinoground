import sys
sys.path.append('./')
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init
import json
from tqdm import tqdm

import argparse

# Create an ArgumentParser object
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument('--data', type=str, default="./Vinoground", help='Path to Vinoground dataset (from Huggingface)')
parser.add_argument('--ckpt', type=str, default="./checkpoints/VideoLlama2", help='Path to model checkpoints')
parser.add_argument("--output", type=str, default="./outputs/videollama2", help="Output directory of score files")

# Parse arguments
args = parser.parse_args()

data_path = args.data
ckpt_path = args.ckpt
output_dir = args.output
import os
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

disable_torch_init()
model_path = ckpt_path
model, processor, tokenizer = model_init(model_path)
modal = "video"

video_ans_file = open(os.path.join(output_dir, "videoscore-response.jsonl"), 'w')
text_ans_file = open(os.path.join(output_dir, "textscore-response.jsonl"), 'w')

with open(os.path.join(data_path, "vinoground_textscore.json"), 'r') as f:
    questions = json.load(f)

for question in tqdm(questions):
    try:
        modal_path = os.path.join(data_path, question["video_name"])
        instruct = question["question"] + "Please only output 1 English character."
        output = mm_infer(processor[modal](modal_path), instruct, model=model, tokenizer=tokenizer, do_sample=False, modal=modal)

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

        modal_path = os.path.join(data_path, question["video_name"])
        instruct = question["question"] + "Please only output 1 English character."
        output = mm_infer(processor[modal](modal_path), instruct, model=model, tokenizer=tokenizer, do_sample=False, modal=modal)

        video_ans_file.write(json.dumps(dict(idx=question["idx"], response=output)) + '\n')
        video_ans_file.flush()
    except Exception as e:
        print(e)
        continue


video_ans_file.close()


