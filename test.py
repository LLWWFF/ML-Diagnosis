import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def load_model(model_path):
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data['weights'], model_data['scaler']

def predict_with_model(X_new, model_path):
    w, scaler = load_model(model_path)
    X_scaled = scaler.transform(X_new)
    X_b = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]
    proba = 1 / (1 + np.exp(-np.dot(X_b, w)))
    return (proba >= 0.5).astype(int)

new_data = pd.read_csv(r"")

features = new_data[['sum_PMC', 'sum_MM*', 'degree', 'ratio_PMC', 'ratio_MM*', 'diff_PMC', 'diff_MM*']]
if 'label' in new_data.columns:
    y_true = new_data['label'].map({'FF': 0, 'F': 1}).values
else:
    raise ValueError("数据文件中缺少 label 列")
predictions = predict_with_model(features, 'trained_logistic_model.pkl')

def gmean(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = recall_score(y_true, y_pred)  # equals tp/(tp+fn)
    return np.sqrt(sensitivity * specificity)
metrics = {
    'Accuracy': accuracy_score(y_true, predictions),
    'Precision': precision_score(y_true, predictions),
    'Recall': recall_score(y_true, predictions),
    'F1-Score': f1_score(y_true, predictions),
    'G-Mean': gmean(y_true, predictions)
}

error_mask = (predictions.flatten() != y_true)
error_data = new_data[error_mask].copy()
error_data['true_label'] = y_true[error_mask]
error_data['predicted_label'] = predictions[error_mask]
error_data['true_label_str'] = error_data['true_label'].map({0: 'FF', 1: 'F'})
error_data['predicted_label_str'] = error_data['predicted_label'].map({0: 'FF', 1: 'F'})
error_types = []
for true, pred in zip(error_data['true_label'], error_data['predicted_label']):
    if true == 0 and pred == 1:
        error_types.append("False Positive (FF→F)")
    elif true == 1 and pred == 0:
        error_types.append("False Negative (F→FF)")
    else:
        error_types.append("Undefined Error")
error_data['error_type'] = error_types
error_data.to_csv("misclassified_samples.csv", index=False)

error_counts = error_data['error_type'].value_counts()
print("\n模型表现评估:")
print(f"Accuracy: {metrics['Accuracy']:.4f}")
print(f"Precision: {metrics['Precision']:.4f}")
print(f"Recall: {metrics['Recall']:.4f}")
print(f"F1-Score: {metrics['F1-Score']:.4f}")
print(f"G-Mean: {metrics['G-Mean']:.4f}\n")

print(f"总错误样本数: {len(error_data)}")
print("错误类型分布:")
print(error_counts.to_string(), "\n")

cm = confusion_matrix(y_true, predictions)
print("混淆矩阵:")
print(pd.DataFrame(cm,
                 index=['True FF (0)', 'True F (1)'],
                 columns=['Predicted FF (0)', 'Predicted F (1)']))

print("\n错误样本已保存至: misclassified_samples.csv")