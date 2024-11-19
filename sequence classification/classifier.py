# %%
import gc
import os
import torch
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm.auto import tqdm
# Suppress warnings
warnings.filterwarnings('ignore')

# %%
embdedding_path = '/mnt/newssd2/amir/Arman/LLM/sequence classification/embedding'
def concatenator(type_):
    items = os.listdir(embdedding_path)
    items.sort()
    embds = torch.tensor([])
    labels = torch.tensor([])
    for item in tqdm(items, desc=f'Load {type_} data'):
        if type_ in item:
            temp_path = os.path.join(embdedding_path, item)
            temp = torch.load(temp_path)
            if 'label' in item:
                labels = torch.concat([labels, temp], dim=0)
            else:
                embds = torch.concat([embds, temp], dim=0)
    
    return np.array(embds), np.array(labels)

# %%
# Load embeddings and labels
x_train, y_train = concatenator('train')
y_train = np.reshape(y_train, (-1, 1))

# %%
x_validation, y_validation = concatenator('validation')
y_validation = np.reshape(y_validation, (-1, 1))

# %%
x_test, y_test = concatenator('test')
y_test = np.reshape(y_test, (-1, 1))

# %%
class LearningRateScheduler():
    def __init__(self, initial_lr, total_steps, warmup_steps):
        self.initial_lr = initial_lr
        self.total_steps = total_steps + 5
        self.warmup_steps = warmup_steps + 5
    
    def get_lr(self, step):
        if step <= self.warmup_steps:
            # Warmup phase: linearly increase learning rate
            return self.initial_lr * (step / self.warmup_steps)
        else:
            # Linear decay phase: linearly decrease learning rate
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return self.initial_lr * (1 - progress)


# %%
# Create DMatrix for the validation set
dvalid = xgb.DMatrix(x_validation, label=y_validation)
# Create DMatrix for the test set
dtest = xgb.DMatrix(x_test, label=y_test)

# %%
# Training configuration
epochs = 32  # Number of epochs
step = 6000 # Step size for incremental training
evals_result = {}
classifier = None  # Placeholder for the classifier
steps = epochs * (len(y_train) // step)  # Total steps for the progress bar
progress_bar = tqdm(range(steps))
num_boost_round = 100  # Number of boosting rounds
# learnin_rate_scheduler = LearningRateScheduler(initial_learning_rate, steps, warmup_steps=)


# Parameters for the XGBoost model
params = {
    'device': 'cuda:1',  # Specify the GPU device
    'tree_method': 'gpu_hist',  # Use GPU for histogram-based algorithm
    'objective': 'binary:logistic',  # Binary classification objective
    'eval_metric': 'logloss',  # Use logloss metric for evaluation
    'max_depth': 100,  # Maximum depth of a tree
    'min_child_weight': 10,  # Minimum sum of instance weight (hessian) needed in a child
    'gamma': 0.65,  # Minimum loss reduction required to make a further partition
    'subsample': 0.8,  # Subsample ratio of training instances
    'colsample_bytree': 0.8,  # Subsample ratio of columns when constructing each tree
    'learning_rate': 0.001,  # Learning rate
    'scale_pos_weight': 3.9,  # Balance of positive and negative weights
    'lambda': 10,  # L2 regularization term
    'alpha': 0.8  # L1 regularization term
}

# %%
# Incremental training loop
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    for i in range(len(y_train) // step):
        dtrain = xgb.DMatrix(x_train[i * step:(i + 1) * step], label=y_train[i * step:(i + 1) * step])
        
        if classifier is None:
            # Train the model for the first time
            classifier = xgb.train(params, dtrain, num_boost_round, evals=[(dtrain, 'train'), (dvalid, 'validation')],
                                   evals_result=evals_result)
        else:
            # Continue training with the existing model
            classifier = xgb.train(params, dtrain, num_boost_round, evals=[(dtrain, 'train'), (dvalid, 'validation')],
                                   evals_result=evals_result, xgb_model=classifier)
        
        progress_bar.update(1)
    
    # Train on the remaining data if it doesn't divide evenly by step
    if len(y_train) % step != 0:
        dtrain = xgb.DMatrix(x_train[(len(y_train) // step) * step:], label=y_train[(len(y_train) // step) * step:])
        classifier = xgb.train(params, dtrain, num_boost_round, evals=[(dtrain, 'train'), (dvalid, 'validation')],
                               evals_result=evals_result, xgb_model=classifier)

# %%
# Save the trained model
classifier.save_model('xgboost_model_incremental.json')

# %%
# Save the training history
torch.save(evals_result, 'xgboost_history')


# %%
# Evaluate the model
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, average_precision_score

# Predict on the training and test sets
preds_train = classifier.predict(dtrain)
preds_validation = classifier.predict(dvalid)
preds_test = classifier.predict(dtest)

# Convert predictions to labels based on threshold 0.5
preds_label_train = [1 if item > 0.5 else 0 for item in preds_train]
preds_label_validation = [1 if item > 0.5 else 0 for item in preds_validation]
preds_label_test = [1 if item > 0.5 else 0 for item in preds_test]

# Test set metrics
accuracy_test = accuracy_score(y_test, preds_label_test)
print(f"Test accuracy: {accuracy_test}")

precision_test = precision_score(y_test, preds_label_test)
print(f"Test Precision: {precision_test}")

recall_test = recall_score(y_test, preds_label_test)
print(f"Test recall_score: {recall_test}")

roc_auc_test = roc_auc_score(y_test, preds_test)
print(f"Test ROC AUC: {roc_auc_test}")

# Validation set metrics
accuracy_validation = accuracy_score(y_validation, preds_label_validation)
print(f"\nValidation accuracy: {accuracy_validation}")

precision_validation = precision_score(y_validation, preds_label_validation)
print(f"Validation Precision: {precision_validation}")

recall_validation = recall_score(y_validation, preds_label_validation)
print(f"Validation recall_score: {recall_validation}")

roc_auc_validation = roc_auc_score(y_validation, preds_validation)
print(f"Validation ROC AUC: {roc_auc_validation}")

# Training set metrics
accuracy_train = accuracy_score(y_train, preds_label_train)
print(f"\nTraining accuracy: {accuracy_train}")

precision_train = precision_score(y_train, preds_label_train)
print(f"Training Precision: {precision_train}")

recall_train = recall_score(y_train, preds_label_train)
print(f"Training recall: {recall_train}")

roc_auc_train = roc_auc_score(y_train, preds_train)
print(f"Training ROC AUC: {roc_auc_train}")

# To plot ROC curve for the test set
import matplotlib.pyplot as plt

fpr, tpr, _ = roc_curve(y_test, preds_test)
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc_test:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Test Set')
plt.legend(loc="lower right")
plt.show()

# %%
