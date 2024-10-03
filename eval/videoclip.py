import torch
from decord import VideoReader, cpu    # pip install decord
from tqdm import tqdm
import cv2
import pandas as pd

from mmpt.models import MMPTModel

def encode_video(video_path, nframe=60):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)
    frame_idx = [i for i in range(0, len(vr), max(1, len(vr) // nframe))]
    frames = vr.get_batch(frame_idx).asnumpy()[:nframe]
    new_frames = [cv2.resize(frames[i], (224, 224)) for i in range(nframe)]
    new_frames = torch.tensor(new_frames, dtype=torch.float32).reshape((1, 2, nframe // 2, 224, 224, 3))
    return new_frames

model, tokenizer, aligner = MMPTModel.from_pretrained("projects/retri/videoclip/how2.yaml")

model.eval()

text_correct = 0
video_correct = 0
group_correct = 0

vino = pd.read_csv("vinoground.csv")
for i in tqdm(range(500)):
    try:
        pos_vid = f"./vinoground_videos/{i}_pos.mp4"
        neg_vid = f"./vinoground_videos/{i}_neg.mp4"
        pos_cap = vino["pos_cap"][i]
        neg_cap = vino["neg_cap"][i]
        # B, T, FPS, H, W, C (VideoCLIP is trained on 30 fps of s3d)
        # video_frames = torch.randn(1, 2, 30, 224, 224, 3)
        pos_frames = encode_video(pos_vid, 32)
        neg_frames = encode_video(neg_vid, 32)

        pos_caps, pos_cmasks = aligner._build_text_seq(
            tokenizer(pos_cap, add_special_tokens=False)["input_ids"]
        )
        neg_caps, neg_cmasks = aligner._build_text_seq(
            tokenizer(neg_cap, add_special_tokens=False)["input_ids"]
        )

        pos_caps, pos_cmasks = pos_caps[None, :], pos_cmasks[None, :]  # bsz=1
        neg_caps, neg_cmasks = neg_caps[None, :], neg_cmasks[None, :]  # bsz=1

        with torch.no_grad():
            output00 = model(pos_frames, pos_caps, pos_cmasks, return_score=True)["score"]
            output01 = model(pos_frames, neg_caps, neg_cmasks, return_score=True)["score"]
            output10 = model(neg_frames, pos_caps, pos_cmasks, return_score=True)["score"]
            output11 = model(neg_frames, neg_caps, neg_cmasks, return_score=True)["score"]
        text_correct += output00 > output01 and output11 > output10
        video_correct += output00 > output10 and output11 > output01
        group_correct += output00 > output01 and output11 > output10 and output00 > output10 and output11 > output01
    except Exception as e:
        print(i, e)
        continue
print(text_correct / 500, video_correct / 500, group_correct / 500)