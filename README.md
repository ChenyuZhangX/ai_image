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

|Metric| Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|
|Full Model|$\bf 0.8580 \uparrow$|$\bf 0.8572 \uparrow$|$\bf 0.8612  \uparrow $|$\bf 0.8618  \uparrow$|
|Simple Model| $0.7839$ | $0.7826$ | $0.7971$ | $0.7879$ |

### Transformer Decoder + Adam + Bigger Feature

- Confusion Matrix

  ![confusion_matrix](./imgs/exp_transformer/confusion_matrix.png)

- Report Metrics



### FC Decoder + Adamx + Smaller Feature

- Confusion Matrix

  ![confusion_matrix](./imgs/exp_fc/confusion_matrix.png)

## Test of CLIP Model and Visualization

For the results please check the [visual.ipynb](./visual.ipynb) for visualization results and clip model outputs.

### CLIP Model

