Code for running vision-language experiments for Understanding the Emergence of Multimodal Representation Alignment. Adapted from [The Platonic Representation Hypothesis](http://arxiv.org/abs/2405.07987).


Install all packages in a new conda environment as follows.
```
conda env create -f platonic.yml
conda activate platonic
```

<hr>

<h3> Running alignment </h3>
<br />

The dataset of unique captions has already been generated: `perturubed/data.json`. To rerun the data generation, first copy `.env.sample` into your own `.env` file with a valid openai API key. Then run:

```bash
python perturbed/create_dataset.py
```

Given the unique captions, we extract features from the models and compute alignment. In `run_platonic_unique.sh`, replace `BASE_DIR` with your own save directory.

```bash
bash run_platonic_unique.sh
```

The resulting features and alignment scores will be stored in `{BASE_DIR}/gpt_u=${perturbation}_feature` and `{BASE_DIR}/gpt_u=${perturbation}_alignment`respectively for varying levels of `perturbation`. Then the results can be plotted as follows.

```
python plot_alignment.py --dataset minhuh/prh --subset wit_1024 --modelset val  --modality_x language --pool_x avg --modality_y vision --pool_y cls --same_scale --features_dir  "{BASE_DIR}/gpt_u={}_features" --align_dir  "{BASE_DIR}/gpt_u={}_alignment" --plot_dir "./plots"
```

<hr>
<h3> Computing alignment-performances scores for  alignment </h3>
<br />

First, download MM-IMDb dataset. Then replace `ROOT` in `mmimdb/constants.py` with the save directory of the dataset. Also, download the indices of the sample data [here](https://drive.google.com/drive/folders/1KXLJDHEFJ9mjT_izfHwmoiy2ROU_t5yb?usp=sharing) and move `train`, `dev`, `test` to `ROOT`.

```bash
wget https://archive.org/download/mmimdb/mmimdb.tar.gz
tar -xvzf mmimdb.tar.gz
rm mmimdb.tar.gz
mv mmimdb /path/to/your/data # Set ROOT to this path
```


To compute alignment-peformance correlation scores for MM-IMDb, run the following.
```bash
bash mmimdb/run_mmimdb.sh
```

<hr>
<h3> Finetuning CLIP </h3>
<br />

Install all packages in a new conda environment as follows.
```
conda env create -f multimodal.yml
conda activate multimodal
```

Finetune CLIP on a given classification task, specified by `class_idx=19`.
```bash
CUDA_VISIBLE_DEVICES=0 \
python mmimdb/clip_finetune_binary.py -m hydra/launcher=ray\
  model.model.clip_weight=0,0.1,0.25,0.5,1.0,2.0,5.0,10.0 \
 +hydra.launcher.ray.max_concurrent_tasks=8 \
  +hydra.launcher.ray.remote.num_cpus=8\
  +hydra.launcher.ray.remote.num_gpus=0.25\
  +model=clip_sup +data=mmimdb \
  mode="train" \
  trainer.max_epochs=30\
  trainer.default_root_dir="/scratch/platonic/mmimdb/test_experiments" \
  +class_idx=19
```


