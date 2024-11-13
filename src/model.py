import torch
import torch.nn as nn
import torch.nn.functional as F


def create_sequential_model(input_dim, hidden_layers=[], dropout_rates=0.3): #Construct a classic model
    layers = []
    last_dim = input_dim
    
    if type(dropout_rates) != list :
        dropout_rates = [dropout_rates]*len(hidden_layers)

    for hidden_dim,dropout_rate in zip(hidden_layers,dropout_rates):
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        last_dim = hidden_dim

    layers.append(nn.Linear(last_dim, 1))
    layers.append(nn.Sigmoid())
    
    model = nn.Sequential(*layers)
    return model

