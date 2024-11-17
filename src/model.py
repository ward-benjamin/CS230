import tensorflow as tf

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


