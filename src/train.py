import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import itertools
import torch
import torch.nn as nn

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


def grid_search_train(X_train, Y_train, X_test, Y_test, model, hyperparameters):


    def generate_values(value):
        """Helper function to generate parameter values (linear or logarithmic)."""
        if isinstance(value, tuple) and value[0] == 'log':  
            start, end, steps = value[1], value[2], value[3]
            return np.logspace(np.log10(start), np.log10(end), num=steps)
        elif isinstance(value,tuple) and value[0] == 'linear':
            start, end, steps = value[1], value[2], value[3]
            return np.linspace(start,end,steps)
        elif isinstance(value, (list, tuple)):  
            return value
        else: 
            return [value]

    keys, values = zip(*hyperparameters.items())
    param_grid = [generate_values(v) for v in values]
    grid_combinations = list(itertools.product(*param_grid))

    results = []

    for combination in grid_combinations:

        params = dict(zip(keys, combination))
        print(f"Training with parameters: {params}")

        batch_size = int(params.get('batch_size', 64)) 
        optimizer = params.get('optimizer', 'adam')
        lr = params.get('learning_rate', 0.0001)
        epochs = params.get('epochs', 20)
        loss = params.get('loss', 'binary_crossentropy')

        if optimizer == "adam":
            optim = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999)
        elif optimizer == 'sgd':
            optim = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
        elif optimizer == 'RMSprop':
            optim = tf.keras.optimizers.RMSprop(learning_rate=lr, rho=0.9)

        model.compile(optimizer=optim, loss=loss, metrics=['accuracy'])

        model.fit(
            X_train, Y_train,
            validation_data=(X_test, Y_test),
            epochs=epochs,
            batch_size=batch_size
        )

        final_test_loss, final_test_accuracy = model.evaluate(X_test, Y_test, verbose=0)

        results.append({'params': params, 'test_loss' : (final_test_loss,final_test_accuracy)})

    return results



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


def train_kan(X_train,Y_train,X_test,Y_test,model,hyperparameters):
    dataset_train = {
        'train_input' : torch.tensor(X_train.values, dtype=torch.float64),
        'train_label' : torch.tensor(Y_train.values, dtype=torch.float64),
        'test_input' : torch.tensor(X_test.values, dtype=torch.float64),
        'test_label' : torch.tensor(Y_test.values, dtype=torch.float64)
    }
    model.speed()
    model.save_act = True
    steps = hyperparameters.get('steps',20)
    l1_rate = hyperparameters.get('l1_rate',0.1)
    lentropy_rate = hyperparameters.get('lentropy_rate',1.0)
    optim = hyperparameters.get('optimizer', 'LBFGS')
    loss = hyperparameters.get('loss_function', nn.BCEWithLogitsLoss())


    result =  model.fit(dataset_train, opt=optim, steps = steps, lamb=0.001, lamb_entropy=lentropy_rate,lamb_l1 = l1_rate,loss_fn = loss)
    return result