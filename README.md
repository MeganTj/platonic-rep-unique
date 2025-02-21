Code for running vision-language experiments for Understanding the Emergence of Multimodal Representation Alignment. Adapted from [The Platonic Representation Hypothesis](http://arxiv.org/abs/2405.07987).


Developed on  

`python = 3.11`
`PyTorch = 2.2.0`

You can install the rest of the requirements via

```bash
pip install -r requirements.txt
```

<hr>

<h3> Running alignment </h3>
<br />

First, we extract features from the models.

```bash
bash extract_unique_features.sh
```

After extracting the features, you can compute the alignment score by 

```bash
bash measure_unique_alignment.sh
```

By default the resulting features and alignment scores will be stored in `/scratch/platonic/results/u=${perturbation}_feature` and `/scratch/platonic/results/u=${perturbation}_alignment`repsectively for varying levels of `perturbation`. Then the results can be plotted as follows.

```
python plot_alignment.py --dataset minhuh/prh --subset wit_1024 --modelset val  --modality_x language --pool_x avg --modality_y vision --pool_y cls --features_dir  "/scratch/platonic/results/u={}_features" --align_dir  "/scratch/platonic/results/u={}_alignment" --plot_dir "./plots"
```


