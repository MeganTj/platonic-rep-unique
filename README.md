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

The resulting alignment scores will be stored in `./results/alignment`


