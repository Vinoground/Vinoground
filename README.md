# Vinoground

We introduce Vinoground, a temporal counterfactual dataset composed of 1000 short and natural video-caption pairs.

This repository hosts the evaluation code of the models we evaluated in our paper, and the instructions on how to reproduce our results. This benchmark has also been integrated into \[[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)\]. One can begin using it by cloning their repository. Evaluation is now made easier. You can also use the evaluation codes we provided in this repository.

\[[Paper](https://arxiv.org/abs/2410.02763)\]\[[Project Page](https://vinoground.github.io/)\]\[[HuggingFace](https://huggingface.co/datasets/HanSolo9682/Vinoground)\]\[[Leaderboard](https://vinoground.github.io/#leaderboard)\]


## Datasets
Please clone our HuggingFace repository, which contains two zipped video folders (`vinoground_videos.zip` and `vinoground_videos_concated.zip`), the text-score and video-score prompts we used (`vinoground_textscore.json` and `vinoground_videoscore.json`) and the original csv file (`vinoground.csv`). In the following instructions, we default that you cloned the Huggingface repo right under this folder. You can use the following code:
```sh
cd Vinoground
unzip vinoground_videos.zip
unzip vinoground_videos_concated.zip
rm -rf *.zip
cd ..
```

## Reproducing Results
Under the folder `eval` lies the code for all the open-source models we evaluated. Of course, prior to reproducing results, you need to properly setup the environment for the models respectively. Once that is done, all the codes have been tidied and can be used with one line of command:
```sh
python eval/xxx.py --data ./Vinoground --ckpt /path/to/ckpt --output ./outputs --nframes 32 --fps 4
```
- `--data` is used to specify the HuggingFace repo's location.
- `--ckpt` is the path to the checkpoints of the model you want to evaluate. Some models like VideoCLIP and ImageBind don't have this option as the checkpoints are small enough to download directly.
- `--output` is used when we evaluate non-CLIP LMMs. The directory stores the corresponding text score and video score results. Use `get_score_all.py` to analyze the results.
- `--nframes` specifies the number of frames you want a model to sample, if supported. Cannot be specified at the same time as `fps`.
- `--fps` specifies the frames per second you want a model to sample, if supported. Cannot be specified at the same time as `nframes`.

Please feel free to evaluate Vinoground on other models. You can use our provided code as templates and modify them according to your model's formats.

## Analyzing Results
We provide code to analyze the results. You can use the following code to produce an Excel sheet that not only contains the overall text, video and group score results but also the categorical results:
```sh
python get_score_all.py --data ./Vinoground --results ./outputs
```
- `--data` is used to specify the HuggingFace repo's location.
- `--results` is the directory that stores the corresponding text score and video score results.

## Citation

If you find Vinoground useful for your research and applications, please cite using this BibTeX:
```bibtex
@article{zhang2024vinoground,
    title={Vinoground: Scrutinizing LMMs over Dense Temporal Reasoning with Short Videos},
    author={Zhang, Jianrui and Mu, Cai and Lee, Yong Jae}
    journal={arXiv},
    year={2024},
    eprint={2410.02763},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2410.02763}, 
}
```