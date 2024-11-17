import tensorflow as tf

def create_sequential_model(input_dim, hidden_layers=[], dropout_rates=0.3, l1_rate=0.01, l2_rate=0.01):  

    model_layers = []
    last_dim = input_dim
    
    if type(dropout_rates) != list:
        dropout_rates = [dropout_rates] * len(hidden_layers)

    for hidden_dim, dropout_rate in zip(hidden_layers, dropout_rates):
        dense_layer = tf.keras.layers.Dense(
            hidden_dim, 
            activation='relu', 
            kernel_initializer=tf.keras.initializers.HeNormal(), 
            bias_initializer=tf.keras.initializers.Zeros(),
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_rate, l2=l2_rate)  
        )
        model_layers.append(dense_layer)
        model_layers.append(tf.keras.layers.Dropout(dropout_rate))
        last_dim = hidden_dim

    final_layer = tf.keras.layers.Dense(
        1, 
        activation='sigmoid', 
        kernel_initializer=tf.keras.initializers.GlorotUniform(), 
        bias_initializer=tf.keras.initializers.Zeros(),
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_rate, l2=l2_rate)  
    )
    model_layers.append(final_layer)

    model = tf.keras.Sequential(model_layers)
    return model


