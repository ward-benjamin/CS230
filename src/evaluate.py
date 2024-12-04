import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch

def evaluate_model(model, X_train,y_train,X_test,y_test):

    """
    In this function, we analyze the results of training a model on its training set and test set.
    """

    # Make predictions on train and test sets
    y_train_pred = model.predict(X_train).round()
    y_test_pred = model.predict(X_test).round()
    
    # Calculate metrics
    Accuracy_train = accuracy_score(y_train, y_train_pred)
    F1_score_train = f1_score(y_train, y_train_pred)
    Accuracy_test = accuracy_score(y_test, y_test_pred)
    F1_score_test = f1_score(y_test, y_test_pred)

    conf_matrix_train = confusion_matrix(y_train, y_train_pred)
    conf_matrix_test = confusion_matrix(y_test, y_test_pred)

    print(f'Accuracy train: {Accuracy_train:.4f}')
    print(f'Accuracy test: {Accuracy_test:.4f}')
    print(f'F1 Score train: {F1_score_train:.4f}')
    print(f'F1 Score test: {F1_score_test:.4f}')   

    #print the heatmap of the confusion matrix
    fig, axes = plt.subplots(1, 2, figsize=(12, 5)) 
    sns.heatmap(conf_matrix_train, annot=True, fmt='.2f', cmap='Blues', cbar=False, ax=axes[0], 
                xticklabels=['0', '1'], yticklabels=['0', '1'])
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_title('Train Confusion Matrix')
    sns.heatmap(conf_matrix_test, annot=True, fmt='.2f', cmap='Blues', cbar=False, ax=axes[1], 
                xticklabels=['0', '1'], yticklabels=['0', '1'])
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    axes[1].set_title('Test Confusion Matrix')
    plt.tight_layout()
    plt.show()


    return {
        'accuracy_train': Accuracy_train,
        'accuracy_test': Accuracy_test,
        'f1_score_train': F1_score_train,
        'f1_score_test': F1_score_test,
        'confusion_matrix_train': conf_matrix_train,
        'confusion_matrix_test': conf_matrix_test
    }


def evaluate_model_kan(X_train,Y_train,X_test,Y_test,model) :
    dataset_test = {
        'train_input' : torch.tensor(X_train.values, dtype=torch.float64),
        'train_label' : torch.tensor(Y_train.values, dtype=torch.float64),
        'test_input' : torch.tensor(X_test.values, dtype=torch.float64),
        'test_label' : torch.tensor(Y_test.values, dtype=torch.float64)
    }
    train_pred = (model(dataset_test['train_input'])>= 0).int()
    test_pred = (model(dataset_test['test_input'])>= 0).int()

    # Calculate metrics
    Accuracy_train = accuracy_score(Y_train, train_pred)
    F1_score_train = f1_score(Y_train, train_pred)
    Accuracy_test = accuracy_score(Y_test, test_pred)
    F1_score_test = f1_score(Y_test, test_pred)

    conf_matrix_train = confusion_matrix(Y_train, train_pred)
    conf_matrix_test = confusion_matrix(Y_test, test_pred)

    print(f'Accuracy train: {Accuracy_train:.4f}')
    print(f'Accuracy test: {Accuracy_test:.4f}')
    print(f'F1 Score train: {F1_score_train:.4f}')
    print(f'F1 Score test: {F1_score_test:.4f}')   

    #print the heatmap of the confusion matrix
    fig, axes = plt.subplots(1, 2, figsize=(12, 5)) 
    sns.heatmap(conf_matrix_train, annot=True, fmt='.2f', cmap='Blues', cbar=False, ax=axes[0], 
                xticklabels=['0', '1'], yticklabels=['0', '1'])
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_title('Train Confusion Matrix')
    sns.heatmap(conf_matrix_test, annot=True, fmt='.2f', cmap='Blues', cbar=False, ax=axes[1], 
                xticklabels=['0', '1'], yticklabels=['0', '1'])
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    axes[1].set_title('Test Confusion Matrix')
    plt.tight_layout()
    plt.show()


    return {
        'accuracy_train': Accuracy_train,
        'accuracy_test': Accuracy_test,
        'f1_score_train': F1_score_train,
        'f1_score_test': F1_score_test,
        'confusion_matrix_train': conf_matrix_train,
        'confusion_matrix_test': conf_matrix_test
    }
