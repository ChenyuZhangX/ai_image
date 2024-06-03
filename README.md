# AI Image Classifiers



## Requirements

Run 

```bash
pip install -r requirements.txt
```

to install the required packages.

## Run

```bash
bash train_xxx.sh # remember to change the config
bash test_xxx.sh # remember to change the config
```

## Results

### Transformer Decoder + Adam + Bigger Feature

- Confusion Matrix

  ![confusion_matrix](./imgs/exp_transformer/confusion_matrix.png)


### FC Decoder + Adamx + Smaller Feature

- Confusion Matrix

  ![confusion_matrix](./imgs/exp_fc/confusion_matrix.png)

## Test of CLIP Model and Visualization

For the results please check the [visual.ipynb](./visual.ipynb) for visualization results and clip model outputs.