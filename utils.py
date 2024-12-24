import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import torch
import os

# 假设 pred_logits 是模型预测的logits
# pred_logits = predictions[0]
# y_true = datasets['test']['labels']  # 真实标签

def softmax(logits):
    exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def compute_metrics(eval_preds):
    labels = eval_preds.label_ids
    preds = eval_preds.predictions.argmax(-1)
    accuracy, precision, recall, f1 = evaluate_model(labels, preds)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# 计算概率和预测标签
def get_predictions(pred_logits):
    probabilities = softmax(pred_logits)
    y_pred = np.argmax(probabilities, axis=1)
    y_pred_score = probabilities[:, 1]  # 第二类的概率
    return y_pred, y_pred_score

# 评估模型
def evaluate_model(y_true, y_pred):
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred)
    return accuracy, precision, recall, f1

def print_evaluation_scores(model_name, y_true, y_pred):
    accuracy, precision, recall, f1 = evaluate_model(y_true, y_pred)
    print(f"{model_name} Model Evaluation")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

# 绘制ROC曲线
if not os.path.exists("figures"):
    os.makedirs("figures")
    print("Directory './figures' created")

def plot_roc_curve(model_name, y_true, y_pred_score):
    y_true, y_pred_score = np.array(y_true), np.array(y_pred_score)
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred_score)
    auc = metrics.auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.savefig(f"figures/roc_{model_name}.png")
    plt.show()
