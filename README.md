<h1 align="center"> Amuse: Human-AI Collaborative Songwriting <br>with Multimodal Inspirations
</h1>

<div align="center">
  <a href="https://yewon-kim.com/" target="_blank">Yewon&nbsp;Kim</a><sup>1</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://sites.google.com/site/wewantsj/" target="_blank">Sung-Ju&nbsp;Lee</a><sup>1</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://chrisdonahue.com/" target="_blank">Chris&nbsp;Donahue</a><sup>2</sup> <br>
  <sup>1</sup>KAIST &emsp; <sup>2</sup>Carnegie Mellon University <br>
</div>
<h3 align="center">[<a href="https://yewon-kim.com/amuse">project page</a>]&emsp;[<a href="http://arxiv.org/abs/2412.18940">arXiv</a>]</h3>
<br>

This repository contains the code for the chord generation method used in <a href="https://yewon-kim.com/amuse">Amuse</a>. If you have any questions, please reach out to <a href="https://yewon-kim.com">Yewon Kim</a> at `yewon.e.kim@kaist.ac.kr`.

## Run
### 1. Environment setup

```bash
conda env create -f environment.yml -n amuse
conda activate amuse
```

### 2. Configure API Key(s)

Create a file `./assets/api_keys.csv` and add your API key(s) in the following format:

| host | key |
| ---- | --- |
| openai | sk-proj-********************************** |

Replace `sk-proj-**********************************` with your OpenAI API key.

### 3. Download Datasets

Download the <a href="https://github.com/chrisdonahue/sheetsage">Hooktheory dataset</a>. For details on the dataset, refer to <a href="https://arxiv.org/abs/2212.01884">this paper</a>.

```bash
cd ./dataset/Hooktheory
wget https://sheetsage.s3.amazonaws.com/hooktheory/Hooktheory.json.gz
```

Download the LLM-generated chords (pre-generated with GPT-4o-2024-05-13):

```bash
cd ./dataset/llmchords
wget https://yewon-kim.com/uploads/publications/2025-amuse/chords.txt
```

### 4. Training
To train a unimodal prior on Hooktheory dataset (P(x) in the <a href="http://arxiv.org/abs/2412.18940">paper</a>), run the following:

```bash
python train.py --dataset hooktheory 
```

To train a unimodal prior on LLM-generated chord progressions (Q(x) in the <a href="http://arxiv.org/abs/2412.18940">paper</a>), run the following:

```bash
python train.py --dataset llmchords 
```

**Note**: If the path specified by `--llmchords_path` (default: `./dataset/llmchords/chords.txt`) does not exist, the above script will generate chords using the ChatGPT API, which incurs API costs. Use the `--llmchords_num` argument to limit API calls. 
By default, the model is trained on pre-generated chords created with GPT-4o-2024-05-13, but please note that this data may be outdated. For reliable chord generation in interactive mode (see section `6: Chord Generation`), we recommend generating new chords using the latest models.

### 5. Evaluation
To compute the similarity between generated chords and the Hooktheory dataset, run:

```bash
python evaluate.py --px_path <path_to_px> --px_step <step_to_load> --qx_path <path_to_qx> --qx_step <step_to_load> 
```

Recommended values for rejection sampling parameters:
* M (`--rej_M`): 4.0-8.0
* Temperature (`--rej_T`): 1.7-2.5

### 6. Chord Generation
To interactively generate chords based on keywords, run:
```bash
python generate.py --px_path <path_to_px> <step_to_load> --qx_path <path_to_qx> --qx_step <step_to_load> --rej_M <threshold_M> --rej_T <temperature>
```
This opens an interactive terminal where you can input keywords and generate chords using two methods: (i) random-sampled LLM (GPT-4o) and rejection-sampled LLM (Amuse) chords.

## Notes

This code may not accurately replicate the results in the paper due to potential inconsistencies during code preparation and changes in OpenAI API versions. For inquiries, please contact <a href="https://yewon-kim.com">Yewon Kim</a> at <a href="mailto:yewon.e.kim@kaist.ac.kr">yewon.e.kim@kaist.ac.kr</a>.

## Citation

If you use this repository in your research, please cite:
```bibtex
@article{kim2024amuse,
    title={Amuse: Human-AI Collaborative Songwriting with Multimodal Inspirations},
    author={Kim, Yewon and Lee, Sung-Ju and Donahue, Chris},
    year={2024},
    journal={arXiv preprint arXiv:2412.18940},
}
```
