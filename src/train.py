import tensorflow as tf
import numpy as np

def train_model(dataset, model, loss_fn, optimizer, num_epochs, batch_size=64, test_ratio=0.05):

    dataset_size = len(dataset)
    train_size = int(dataset_size * (1 - test_ratio))
    test_size = dataset_size - train_size

    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)


    train_dataset_batch = train_dataset.shuffle(buffer_size=1000).batch(batch_size)
    test_dataset_batch = test_dataset.batch(test_size)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        train_dataset_batch,
        validation_data=test_dataset_batch,
        epochs=num_epochs
    )

    return history,test_dataset
