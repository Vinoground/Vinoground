from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from tqdm import tqdm

import sys
if "nframes" in sys.argv and "fps" in sys.argv:
    raise ValueError("You can only specify one: nframes or fps.")
import argparse

# Create an ArgumentParser object
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument('--data', type=str, default="./Vinoground", help='Path to Vinoground dataset (from Huggingface)')
parser.add_argument('--ckpt', type=str, default="./checkpoints/qwen", help='Path to model checkpoints')
parser.add_argument("--output", type=str, default="./outputs", help="Output directory of score files")
parser.add_argument("--nframes", type=int, default=32, help="Number of frames to sample.")
parser.add_argument("--fps", type=int, default=2, help="Frames per second to sample frames.")

# Parse arguments
args = parser.parse_args()

data_path = args.data
ckpt_path = args.ckpt
output_dir = args.output
nframes = args.nframes
fps = args.fps

import os
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model_name = ckpt_path

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# default processer
processor = AutoProcessor.from_pretrained(model_name)

if "fps" in sys.argv:
    os.mkdir(os.path.join(output_dir, f"{model_name}-fps{fps}"))
    video_ans_file = open(os.path.join(output_dir, f"{model_name}-fps{fps}", f"videoscore-response.jsonl"), 'w')
    text_ans_file = open(os.path.join(output_dir, f"{model_name}-fps{fps}", f"textscore-response.jsonl"), 'w')
else:
    os.mkdir(os.path.join(output_dir, f"{model_name}-frame{nframes}"))
    video_ans_file = open(os.path.join(output_dir, f"{model_name}-frame{nframes}", f"videoscore-response.jsonl"), 'w')
    text_ans_file = open(os.path.join(output_dir, f"{model_name}-frame{nframes}", f"textscore-frame{nframes}-response.jsonl"), 'w')

with open(os.path.join(data_path, "vinoground_textscore.json"), 'r') as f:
    questions = json.load(f)

for question in tqdm(questions):
    try:

        if "fps" in sys.argv:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": os.path.join(data_path, question["video_name"]),
                            "fps": fps
                        },
                        {
                            "type": "text",
                            "text": question["question"],
                        },
                    ],
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": os.path.join(data_path, question["video_name"]),
                            "nframes": nframes
                        },
                        {
                            "type": "text",
                            "text": question["question"],
                        },
                    ],
                }
            ]

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        text_ans_file.write(json.dumps(dict(idx=question["idx"], response=output_text)) + '\n')
        text_ans_file.flush()
    except Exception as e:
        print(e)
        continue


text_ans_file.close()


with open(os.path.join(data_path, "vinoground_videoscore.json"), 'r') as f:
    questions = json.load(f)

for question in tqdm(questions):
    try:
        if "fps" in sys.argv:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": os.path.join(data_path, question["video_name"]),
                            "fps": fps
                        },
                        {
                            "type": "text",
                            "text": question["question"],
                        },
                    ],
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": os.path.join(data_path, question["video_name"]),
                            "nframes": nframes
                        },
                        {
                            "type": "text",
                            "text": question["question"],
                        },
                    ],
                }
            ]


        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        video_ans_file.write(json.dumps(dict(idx=question["idx"], response=output_text)) + '\n')
        video_ans_file.flush()
    except Exception as e:
        print(e)
        continue


video_ans_file.close()