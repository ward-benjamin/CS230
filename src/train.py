import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

"""
In this file, we gather the functions necessary to fit the models used.
"""

def train_model(X_train,Y_train,X_test,Y_test,model,hyperparameters):

    batch_size = hyperparameters.get('batch_size',64) #batch size 
    optimizer = hyperparameters.get('optimizer','adam') #optimzer between sgd,RMSprop, or adam
    lr = hyperparameters.get('learning_rate', 0.0001) #learning rate
    epoch = hyperparameters.get('epochs', 20) #number of epochs
    loss = hyperparameters.get('loss','binary_accuracy') #choice of the loss
    if optimizer == "adam":
        optim = tf.keras.optimizers.Adam(learning_rate=lr,beta_1=0.9, beta_2=0.999)
    elif optimizer == 'sgd':
        optim = tf.keras.optimizers.SGD(learning_rate=lr,momentum=0.9)
    elif optimizer == 'RMSprop':
        optim = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)


    model.compile(
        optimizer=optim,
        loss=loss,
        metrics=['accuracy']
    )

    history = model.fit(
        X_train,Y_train,
        validation_data=(X_test,Y_test),
        epochs=epoch,
        batch_size = batch_size
    )

    return history



def train_logistic(X_train,Y_train,X_test,Y_test,hyperparameters):
    penality = hyperparameters.get('penality',None) #Name of the penality
    solver = hyperparameters.get('solver','lbfgs') #solver
    max_iter = hyperparameters.get('epoch',1000) #number of epochs
    
    scaler = StandardScaler()
    X_train_transform = scaler.fit_transform(X_train)
    X_test_transform = scaler.transform(X_test)

    model = LogisticRegression(
        penalty=penality,         
        solver=solver,        
        max_iter=max_iter,           
        tol=1e-6,               
        fit_intercept=True,     
        random_state=42         
    )

    model.fit(X_train_transform, Y_train)

    return model,X_train_transform,X_test_transform
