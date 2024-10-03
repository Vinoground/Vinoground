from PIL import Image
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor 
from decord import VideoReader, cpu    # pip install decord
from tqdm import tqdm
import json
import argparse

# Create an ArgumentParser object
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument('--data', type=str, default="./Vinoground", help='Path to Vinoground dataset (from Huggingface)')
parser.add_argument('--ckpt', type=str, default="./checkpoints/Phi-3-5-Vision", help='Path to model checkpoints')
parser.add_argument("--output", type=str, default="./outputs/phi35vision", help="Output directory of score files")
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

model_id = ckpt_path 

# Note: set _attn_implementation='eager' if you don't have flash_attn installed
model = AutoModelForCausalLM.from_pretrained(
  model_id, 
  device_map="cuda", 
  trust_remote_code=True, 
  torch_dtype="auto", 
  _attn_implementation='flash_attention_2'    
)

# for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
processor = AutoProcessor.from_pretrained("microsoft/Phi-3.5-vision-instruct", 
  trust_remote_code=True, 
  num_crops=4
) 


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


video_ans_file = open(os.path.join(output_dir, f"videoscore-frame{nframes}-response.jsonl"), 'w')
text_ans_file = open(os.path.join(output_dir, f"textscore-frame{nframes}-response.jsonl"), 'w')

with open(os.path.join(data_path, "vinoground_textscore.json"), 'r') as f:
    questions = json.load(f)

for question in tqdm(questions):
    try:

        frames = encode_video(os.path.join(data_path, question["video_name"]))
        q = ""
        for i in range(len(frames)):
            q += f"<|image_{i+1}|>\n"
        messages = [
            {
                'role': 'user', 
                'content': q + question["question"] + "Please only output one English character."
            }, 
        ]


        prompt = processor.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        inputs = processor(prompt, frames, return_tensors="pt").to("cuda:0") 

        generation_args = { 
            "max_new_tokens": 1000, 
            "temperature": 0.0, 
            "do_sample": False, 
        } 

        generate_ids = model.generate(**inputs, 
            eos_token_id=processor.tokenizer.eos_token_id, 
            **generation_args
        )

        # remove input tokens 
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = processor.batch_decode(generate_ids, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False)[0] 

        
        text_ans_file.write(json.dumps(dict(idx=question["idx"], response=response)) + '\n')
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
        q = ""
        for i in range(len(frames)):
            q += f"<|image_{i+1}|>\n"
        messages = [
            {
                'role': 'user', 
                'content': q + question["question"] + "Please only output one English character."
            }, 
        ]


        prompt = processor.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        inputs = processor(prompt, frames, return_tensors="pt").to("cuda:0") 

        generation_args = { 
            "max_new_tokens": 1000, 
            "temperature": 0.0, 
            "do_sample": False, 
        } 

        generate_ids = model.generate(**inputs, 
            eos_token_id=processor.tokenizer.eos_token_id, 
            **generation_args
        )

        # remove input tokens 
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = processor.batch_decode(generate_ids, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False)[0] 
        
        video_ans_file.write(json.dumps(dict(idx=question["idx"], response=response)) + '\n')
        video_ans_file.flush()

    except Exception as e:
        print(e)
        continue


video_ans_file.close()