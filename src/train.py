import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def train_model(X_train,Y_train,X_test,Y_test,model,hyperparameters):

    batch_size = hyperparameters.get('batch_size',64)
    optimizer = hyperparameters.get('optimizer','adam')
    lr = hyperparameters.get('learning_rate', 0.0001)
    epoch = hyperparameters.get('epochs', 20)
    loss = hyperparameters.get('loss','binary_accuracy')
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
    penality = hyperparameters.get('penality',None)
    solver = hyperparameters.get('solver','lbfgs')
    max_iter = hyperparameters.get('epoch',1000)
    
    scaler = StandardScaler()
    X_train_transform = scaler.fit_transform(X_train)
    X_test_transform = scaler.transform(X_test)
    # Create the unregularized logistic regression model
    model = LogisticRegression(
        penalty=penality,         # No regularization
        solver=solver,         # Optimization solver
        max_iter=max_iter,           # Maximum number of iterations
        tol=1e-6,               # Tolerance for stopping criteria
        fit_intercept=True,     # Include intercept term
        random_state=42         # Ensure reproducibility
    )
    # Train the model on the training data
    model.fit(X_train_transform, Y_train)

    return model,X_train_transform,X_test_transform
