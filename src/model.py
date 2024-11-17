import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.initializers import HeNormal, Zeros, GlorotUniform

def create_sequential_model(input_dim, hidden_layers=[], dropout_rates=0.3):  

    model_layers = []
    last_dim = input_dim
    
    if type(dropout_rates) != list:
        dropout_rates = [dropout_rates] * len(hidden_layers)

    for hidden_dim, dropout_rate in zip(hidden_layers, dropout_rates):
        dense_layer = layers.Dense(
            hidden_dim, 
            activation='relu', 
            kernel_initializer=HeNormal(), 
            bias_initializer=Zeros()
        )
        model_layers.append(dense_layer)
        model_layers.append(layers.Dropout(dropout_rate))
        last_dim = hidden_dim

    final_layer = layers.Dense(
        1, 
        activation='sigmoid', 
        kernel_initializer=GlorotUniform(), 
        bias_initializer=Zeros()
    )
    model_layers.append(final_layer)

    model = Sequential(model_layers)
    return model

