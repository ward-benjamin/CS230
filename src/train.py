import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.optim as optim

def train_model(dataset,model,criterion,optimizer,num_epochs,batch_size = 64,test_ratio = 0.05) :
    train_size = int(len(dataset) * test_ratio)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    train_losses = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()  
        running_loss = 0.0
        
        # Train over batches in the train loader
        for batch_features, batch_targets in train_loader:
            optimizer.zero_grad()  
            outputs = model(batch_features)  
            loss = criterion(outputs, batch_targets)  
            loss.backward()  
            optimizer.step() 
            
            running_loss += loss.item()
            train_losses.append(loss.item())
        
        avg_loss = running_loss / len(train_loader)  
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')


    model.eval()  
    correct_train = 0
    total_train = 0
    with torch.no_grad():  
        for batch_features, batch_targets in train_loader:
            outputs = model(batch_features)
            predicted = (outputs > 0.5).float()  
            correct_train += (predicted == batch_targets).sum().item()
            total_train += batch_targets.size(0)
        
    train_accuracy = correct_train / total_train  
    
    
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for batch_features, batch_targets in test_loader:
            outputs = model(batch_features)
            predicted = (outputs > 0.5).float()  
            correct_test += (predicted == batch_targets).sum().item()
            total_test += batch_targets.size(0)
    
    test_accuracy = correct_test / total_test 

    print(f'Final Train Accuracy: {train_accuracy:.4f}')
    print(f'Final Test Accuracy: {test_accuracy:.4f}')

    return train_losses