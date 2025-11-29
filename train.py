import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle 

data = pd.read_csv('data/dataset-new.csv')
data['label'] = data['label'].map({'FF': 0, 'F': 1})
y = data['label'].values.reshape(-1, 1)
features = data[['sum_PMC', 'sum_MM*', 'degree', 'ratio_PMC', 'ratio_MM*', 'diff_PMC', 'diff_MM*']]
X = features
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_train_final = np.hstack((np.ones((X_train_scaled.shape[0], 1)), X_train_scaled))
X_val_final = np.hstack((np.ones((X_val_scaled.shape[0], 1)), X_val_scaled))
np.random.seed(42)
w = np.random.randn(X_train_final.shape[1], 1)
learning_rate = 0.1
iterations = 1001
train_losses = []
val_losses = []
iterations_list = []
for i in range(1,iterations):
    z = np.dot(X_train_final, w)
    h = 1 / (1 + np.exp(-z))
    loss = -np.mean(y_train * np.log(h) + (1 - y_train) * np.log(1 - h))
    gradient = np.dot(X_train_final.T, (h - y_train)) / len(y_train)
    w -= learning_rate * gradient
    if i % 20 == 0 or i == 1:
        z_val = np.dot(X_val_final, w)
        h_val = 1 / (1 + np.exp(-z_val))
        val_loss = -np.mean(y_val * np.log(h_val) + (1 - y_val) * np.log(1 - h_val))
        y_val_pred = (h_val >= 0.5).astype(int)
        train_losses.append(loss)
        val_losses.append(val_loss)
        iterations_list.append(i)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        print(f'Iteration {i}, Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
loss_df = pd.DataFrame({
    'Iteration': iterations_list,
    'Train Loss': train_losses,
    'Validation Loss': val_losses
})
loss_df.to_csv('training_loss_data.csv', index=False)
plt.figure(figsize=(10, 6))
plt.plot(iterations_list, train_losses, 'b-o', label='Train Loss')
plt.plot(iterations_list, val_losses, 'r--s', label='Val Loss')
plt.xlabel('Iterations', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training & Validation Loss Curve', fontsize=14)
plt.legend()
plt.grid(linestyle='--', alpha=0.7)
plt.show()

def predict(X_new, scaler, w):
    X_scaled = scaler.transform(X_new)
    X_b = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]
    proba = 1 / (1 + np.exp(-np.dot(X_b, w)))
    return (proba >= 0.5).astype(int)
y_val_pred = predict(X_val, scaler, w)
cm = confusion_matrix(y_val, y_val_pred)
cm_df = pd.DataFrame(cm,
                    index=['True FF (0)', 'True F (1)'],
                    columns=['Predicted FF (0)', 'Predicted F (1)'])
cm_df.to_csv('confusion_matrix.csv')

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['FF (0)', 'F (1)'],
            yticklabels=['FF (0)', 'F (1)'])
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.title('Confusion Matrix', fontsize=14)
plt.show()
feature_names = features.columns.tolist()
weights = w.ravel()[1:]  # 排除偏置项
importance = np.abs(weights)
sorted_idx = np.argsort(importance)
sorted_features = [feature_names[i] for i in sorted_idx]
sorted_importance = importance[sorted_idx]
importance_df = pd.DataFrame({
    'Feature': sorted_features,
    'Absolute Importance': sorted_importance
})
importance_df.to_csv('feature_importance.csv', index=False)

plt.figure(figsize=(10, 6))
plt.barh(sorted_features, sorted_importance, color='steelblue')
plt.xlabel('Absolute Weight Importance', fontsize=12)
plt.title('Feature Importance Ranking', fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
accuracy = accuracy_score(y_val, y_val_pred)
precision = precision_score(y_val, y_val_pred)
recall = recall_score(y_val, y_val_pred)
f1 = f1_score(y_val, y_val_pred)

metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Value': [accuracy, precision, recall, f1]
})
metrics_df.to_csv('performance_metrics.csv', index=False)



print('\n验证集评估指标:')
print(f'准确度 (Accuracy): {accuracy:.4f}')
print(f'精确度 (Precision): {precision:.4f}')
print(f'召回率 (Recall): {recall:.4f}')
print(f'F1值 (F1 Score): {f1:.4f}')
model_data = {
    'weights': w,
    'scaler': scaler
}

with open('trained_logistic_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("模型已保存为 trained_logistic_model.pkl")