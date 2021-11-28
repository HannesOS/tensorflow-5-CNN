import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_data():
    """
    Loading and preprocessing the data.
        Results:
            train_ds,test_ds,valid_ds: the preprocessed datasets
    """

    # load data from tfds
    train_ds, valid_ds, test_ds = tfds.load(name="fashion_mnist", split=['train+test[:80]','train+test[80:90]', 'train+test[90:]'], as_supervised=True)

    # apply preprocessing to the datasets
    train_ds = prepare_data(train_ds)
    valid_ds = prepare_data(valid_ds)
    test_ds = prepare_data(test_ds)

    return train_ds, valid_ds, test_ds

def prepare_data(ds):
    """
    Preparing our data for our model.
        Args:
        ds: the dataset we want to preprocess
        Results:
        ds: preprocessed dataset
    """
    #casting each element to tf.float32
    ds = ds.map(lambda feature, target: (tf.cast(feature, tf.float32), target))
    #ds = ds.map(lambda feature, target: (feature, tf.cast(target, tf.float32)))
    ds = ds.map(lambda feature, target: (feature, tf.one_hot(target, depth=10)))
    #
    #ds = ds.map(lambda feature, target: (tf.cast(feature, tf.float32), tf.cast(target, tf.float32)))

    #
    ds = ds.cache()
    # shuffle, batch, prefetch our dataset
    ds = ds.shuffle(10000)
    ds = ds.batch(32)
    ds = ds.prefetch(20)
    return ds


def train_step(model, input, target, loss_function, optimizer):
  """
  Performs a forward and backward pass for  one dataponit of our training set
    Args:
      model: our created MLP model (MyModel object)
      input: our input (tensor)
      target: our target (tensor)
      loss_funcion: function we used for calculating our loss (keras function)
      optimizer: our optimizer used for packpropagation (keras function)
    Results:
      loss: our calculated loss for the datapoint (float)
    """

  with tf.GradientTape() as tape:

    # forward step
    prediction = model(input)

    # calculating loss
    loss = loss_function(target, prediction)

    # calculaing the gradients
    gradients = tape.gradient(loss, model.trainable_variables)

  # updating weights and biases
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  return loss


def test(model, test_data, loss_function):
  """
  Test our MLP, by going through our testing dataset,
  performing a forward pass and calculating loss and accuracy
    Args:
      model: our created MLP model (MyModel object)
      test_data: our preprocessed test dataset (set of tuples with tensors)
      loss_funcion: function we used for calculating our loss (keras function)
    Results:
        loss: our mean loss for this epoch (float)
        accuracy: our mean accuracy for this epoch (float)
  """

  # initializing lists for accuracys and loss
  accuracy_aggregator = []
  loss_aggregator = []

  for (input, target) in test_data:

    # forward step
    prediction = model(input)

    # calculating loss
    loss = loss_function(target, prediction)

    # calculating accuracy
    accuracy =  np.argmax(target.numpy(), axis=1) == np.argmax(prediction.numpy(), axis=1)
    accuracy = np.mean(accuracy)

    # add loss and accuracy to the lists
    loss_aggregator.append(loss.numpy())
    accuracy_aggregator.append(np.mean(accuracy))

  # calculate the mean of the loss and accuracy (for this epoch)
  loss = tf.reduce_mean(loss_aggregator)
  accuracy = tf.reduce_mean(accuracy_aggregator)

  return loss, accuracy


def visualize(train_losses, valid_losses, valid_accuracies):
    """
    Displays the losses and accuracies from the different models in a plot-grid.
    Args:
      train_losses = mean training losses per epoch
      valid_losses = mean testing losses per epoch
      valid_accuracies = mean accuracies (testing dataset) per epoch
    """

    fig, ax = plt.subplots(2,1)
    ax[0].plot(train_losses)
    ax[0].plot(valid_losses)
    ax[1].plot(valid_accuracies)

    fig.legend(["Train_ds loss", " Valid_ds loss", " Valid_ds accuracy"])
    plt.xlabel("Training epoch")
    fig.tight_layout()