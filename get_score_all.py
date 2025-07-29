import json
import os
import csv
import openpyxl
import argparse

# Create an ArgumentParser object
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument('--data', type=str, default="./Vinoground", help='Path to Vinoground dataset (from Huggingface)')
parser.add_argument('--results', type=str, default="./outputs", help="The folder where all evaluation results are stored.")

# Parse arguments
args = parser.parse_args()

data_path = args.data
results = args.results
models = os.listdir(results)


wb = openpyxl.Workbook()
ws = wb.active
ws.title = "all"
ws.append(["model", "text", "video", "group"])


text_json_file = os.path.join(data_path, "vinoground_textscore.json")
video_json_file = os.path.join(data_path, "vinoground_videoscore.json")
with open(video_json_file, 'r') as f:
    video_gt_data = json.load(f)

with open(text_json_file, 'r') as f:
    text_gt_data = json.load(f)

with open(os.path.join(data_path, "vinoground.csv"), 'r') as f:
    vino = list(csv.reader(f))[1:]


for model in models:
    category_all = {}
    category_text_correct = {}
    category_video_correct = {}
    category_group_correct = {}
    try:
        text_pred_json_file = os.path.join(results, model, "textscore-response.jsonl")
        with open(text_pred_json_file, 'r') as f:
            text_pred_data = [json.loads(line) for line in f]
        
        text_pred2content = {}
        for text_pred_item in text_pred_data:
            text_pred2content[text_pred_item['idx']] = text_pred_item['response']

        
        video_pred_json_file = os.path.join(results, model, "videoscore-response.jsonl")
        with open(video_pred_json_file, 'r') as f:
            video_pred_data = [json.loads(line) for line in f]
        
        video_pred2content = {}
        for video_pred_item in video_pred_data:
            video_pred2content[video_pred_item['idx']] = video_pred_item['response']
                        
                        
        def get_text_correct_wrong(item):
            gt = item['GT']
            if "qwen" or "ma-lmm" in model:
                if text_pred2content[item['idx']][0][:len(gt)].lower() == gt.lower():
                    return True
                else:
                    return False
            else:
                if text_pred2content[item['idx']][:len(gt)].lower() == gt.lower():
                    return True
                else:
                    return False
            
        def get_video_correct_wrong(item):
            gt = item['GT']
            if "qwen" or "ma-lmm" in model:
                if video_pred2content[item['idx']][0][:len(gt)].lower() == gt.lower():
                    return True
                else:
                    return False
            else:
                if video_pred2content[item['idx']][:len(gt)].lower() == gt.lower():
                    return True
                else:
                    return False

        text_gt_pair_data = [ [text_gt_data[2*i], text_gt_data[2*i+1] ] for i in range(len(text_gt_data)//2)]
        video_gt_pair_data = [ [video_gt_data[2*i], video_gt_data[2*i+1] ] for i in range(len(video_gt_data)//2)]

        for i in range(len(text_gt_pair_data)):
            try:
                categories = ["all", vino[i][1]]
                categories.extend(vino[i][2].split(';'))
                if categories[-1] == "":
                    categories.pop(-1)
                for cat in categories:
                    if cat not in category_all.keys():
                        category_all[cat] = 0
                        category_text_correct[cat] = 0
                        category_video_correct[cat] = 0
                        category_group_correct[cat] = 0
                is_curr_right = True
                pos, neg = text_gt_pair_data[i]

                for cat in categories:
                    category_all[cat] += 1
                if get_text_correct_wrong(pos) and get_text_correct_wrong(neg):
                    for cat in categories:
                        category_text_correct[cat] += 1
                else:  
                    is_curr_right = False
            
                pos, neg = video_gt_pair_data[i]
                if get_video_correct_wrong(pos) and get_video_correct_wrong(neg):
                    for cat in categories:
                        category_video_correct[cat] += 1
                else:
                    is_curr_right = False
                
                if is_curr_right:
                    for cat in categories:
                        category_group_correct[cat] += 1
            except:
                continue

        ws.append([model, f"{category_text_correct['all']/category_all['all']*100:.2f}", f"{category_video_correct['all']/category_all['all']*100:.2f}", f"{category_group_correct['all']/category_all['all']*100:.2f}"])
        ws2 = wb.create_sheet(title=model.replace(":", ""))

        ws2.append(["category", "text", "video", "group"])
        for cat in category_all.keys():
            ws2.append([cat, f"{category_text_correct[cat]/category_all[cat]*100:.2f}", f"{category_video_correct[cat]/category_all[cat]*100:.2f}", f"{category_group_correct[cat]/category_all[cat]*100:.2f}"])
            
    except Exception as e:
        continue

wb.save("vinoground_evaluation_results.xlsx")
