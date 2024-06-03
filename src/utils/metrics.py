from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import os

def get_metrics(preds, probs, gts, save_path):
    '''
    calculate the metrics and save

    preds: numpy array of shape (N,)
    probs: numpy array of shape (N, num_classes)
    gts: numpy array of shape (N,)
    '''
    acc = accuracy_score(gts, preds)
    f1 = f1_score(gts, preds, average='macro')
    precision = precision_score(gts, preds, average='macro')
    recall = recall_score(gts, preds, average='macro')

    os.makedirs(save_path, exist_ok=True)

    # confusion matrix
    num_classes = probs.shape[1]
    cm = confusion_matrix(gts, preds)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.xticks(range(num_classes), range(num_classes))
    plt.yticks(range(num_classes), range(num_classes))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))

    # save metrics
    with open(os.path.join(save_path, 'metrics.txt'), 'w') as f:
        f.write(f"Accuracy: {acc}\n")
        f.write(f"F1: {f1}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")