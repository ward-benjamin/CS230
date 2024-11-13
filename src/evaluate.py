import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, dev, device='cpu'):
    model.eval() 


    dev_features, dev_targets = dev.dataset[dev.indices]
    dev_features, dev_targets = dev_features.to(device), dev_targets.to(device)


    with torch.no_grad():
        outputs = model(dev_features)
        predicted = (outputs > 0.5).float()  


    accuracy = accuracy_score(dev_targets.cpu(), predicted.cpu())
    f1 = f1_score(dev_targets.cpu(), predicted.cpu())

    cm = confusion_matrix(dev_targets.cpu(), predicted.cpu())
    
    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1 Score: {f1:.4f}')

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['0', '1'], yticklabels=['0', '1'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'false_positive': cm[0, 1],
        'false_negative': cm[1, 0]
    }