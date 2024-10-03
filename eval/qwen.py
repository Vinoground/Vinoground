from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from tqdm import tqdm

model_name = "Qwen/Qwen2-VL-72B-Instruct"

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# default processer
processor = AutoProcessor.from_pretrained(model_name)


for nframes in [32,16,8,4,2]:
# for fps in [4,2,1,0.5]:
    video_ans_file = open(f"qwen2-vl-72b-videoscore-frame{nframes}-response.jsonl", 'w')
    text_ans_file = open(f"qwen2-vl-72b-textscore-frame{nframes}-response.jsonl", 'w')
    # video_ans_file = open(f"qwen2-vl-7b-CoT-videoscore-fps{fps}-response.jsonl", 'w')
    # text_ans_file = open(f"qwen2-vl-7b-CoT-textscore-fps{fps}-response.jsonl", 'w')

    with open("vinoground_textscore.json", 'r') as f:
        questions = json.load(f)

    for question in tqdm(questions):
        try:

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": question["video_name"],
                            "nframes": nframes
                            # "fps": fps
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


    with open("vinoground_videoscore.json", 'r') as f:
        questions = json.load(f)

    for question in tqdm(questions):
        try:
            # messages = [
            #     {
            #         "role": "user",
            #         "content": [
            #             {
            #                 "type": "text",
            #                 "text": "This is the first video:",
            #             },
            #             {
            #                 "type": "video",
            #                 "video": "vinoground_videos/" + question["vid1"],
            #                 "nframes": nframes,
            #             },
            #             {
            #                 "type": "text",
            #                 "text": "This is the second video:",
            #             },
            #             {
            #                 "type": "video",
            #                 "video": "vinoground_videos/" + question["vid2"],
            #                 "nframes": nframes,
            #             },
            #             {
            #                 "type": "text",
            #                 "text": question["question"] + "Please only output one English character.",
            #             },
            #         ],
            #     }
            # ]
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": question["video_name"],
                            "nframes": nframes,
                            # "fps": fps
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