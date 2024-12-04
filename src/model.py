import tensorflow as tf
import itertools
from kan.MultKAN import KAN
import torch

"""
This file contains the methods to build the different models used in the project.
"""

def create_sequential_model(input_dim, hidden_layers=[], dropout_rates=0.3, l1_rate=0.01, l2_rate=0.01):  
    """
    This function builds a dense model with TensorFlow, 
    using ReLU as the activation function and sigmoid for the output. 
    Furthermore, we add dropout and l1/L2 regularization layers.
    """

    model_layers = []

    
    if type(dropout_rates) != list:
        dropout_rates = [dropout_rates] * len(hidden_layers)

    for hidden_dim, dropout_rate in zip(hidden_layers, dropout_rates):
        dense_layer = tf.keras.layers.Dense(
            hidden_dim, 
            activation='relu', 
            kernel_initializer=tf.keras.initializers.HeNormal(), #We initiate this layer with He, because the activation function is relu
            bias_initializer=tf.keras.initializers.Zeros(),
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_rate, l2=l2_rate)  
        )
        model_layers.append(dense_layer)
        model_layers.append(tf.keras.layers.Dropout(dropout_rate))


    final_layer = tf.keras.layers.Dense(
        1, 
        activation='sigmoid', 
        kernel_initializer=tf.keras.initializers.GlorotUniform(), #We initiate this layer with Xavier, because the activation function is sigmoid
        bias_initializer=tf.keras.initializers.Zeros(),
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_rate, l2=l2_rate)  
    )
    model_layers.append(final_layer)

    model = tf.keras.Sequential(model_layers)
    return model


def create_list_model(input_dim,hidden_layers_list,dropout_rates_list,l1_rate_list,l2_rate_list):
    final_list = []
    for hidden_layers, dropout_rates,l1_rate,l2_rate in itertools.product(hidden_layers_list,dropout_rates_list,l1_rate_list,l2_rate_list):
        model = create_sequential_model(input_dim,hidden_layers = hidden_layers,dropout_rates=dropout_rates,l1_rate=l1_rate,l2_rate=l2_rate)
        final_list.append(model)
    return final_list



def create_kan_model(input_dim, hidden_layers,grid = 5, degree = 3):
    torch.set_default_dtype(torch.float64)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_0 = KAN(width = [input_dim]+hidden_layers+[1],grid = grid, k = degree, device=device)
    return model_0

