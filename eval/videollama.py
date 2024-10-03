import sys
sys.path.append('./')
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init
import json
from tqdm import tqdm


disable_torch_init()
model_path = 'VideoLLaMA2-72B'
model, processor, tokenizer = model_init(model_path)
modal = "video"

video_ans_file = open(f"videollama2-72b-videoscore-response.jsonl", 'w')
text_ans_file = open(f"videollama2-72b-textscore-response.jsonl", 'w')

with open("vinoground_textscore.json", 'r') as f:
    questions = json.load(f)

for question in tqdm(questions):
    try:
        modal_path = "../" + question["video_name"]
        instruct = question["question"] + "Please only output 1 English character."
        output = mm_infer(processor[modal](modal_path), instruct, model=model, tokenizer=tokenizer, do_sample=False, modal=modal)

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

        modal_path = "../" + question["video_name"]
        instruct = question["question"] + "Please only output 1 English character."
        output = mm_infer(processor[modal](modal_path), instruct, model=model, tokenizer=tokenizer, do_sample=False, modal=modal)

        video_ans_file.write(json.dumps(dict(idx=question["idx"], response=output)) + '\n')
        video_ans_file.flush()
    except Exception as e:
        print(e)
        continue


video_ans_file.close()


