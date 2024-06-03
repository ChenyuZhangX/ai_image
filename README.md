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


## 1. Main Results

|Metric| Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|
|Full Model|$\bf 0.8580 \uparrow$|$\bf 0.8572 \uparrow$|$\bf 0.8612  \uparrow $|$\bf 0.8618  \uparrow$|
|Simple Model| $0.7839$ | $0.7826$ | $0.7971$ | $0.7879$ |
|CLIP Model | $0.3698$ | $0.3454$ | $0.3203$ | $0.3784$ |

### 1.1. Transformer Decoder + Adam + Bigger Feature

- Confusion Matrix

  ![confusion_matrix](./imgs/exp_transformer/confusion_matrix.png)

- Report Metrics



### 1.2. FC Decoder + Adamx + Smaller Feature

- Confusion Matrix

  ![confusion_matrix](./imgs/exp_fc/confusion_matrix.png)


### 1.3. CLIP Model

**For the results please check the [visual.ipynb](./visual.ipynb) for visualization results and clip model outputs.**


- Confusion Matrix

  ![confusion_matrix](./imgs/clip/confusion_matrix.png)

## 2. Ablation for Full Model

Use 
