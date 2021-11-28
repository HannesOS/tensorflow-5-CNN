import tensorflow as tf

class MyModel(tf.keras.Model):
  """
  Our own custon MLP model, which inherits from the keras.Model class
    Functions:
      init: constructor of our model
      call: performs forward pass of our model
  """

  def __init__(self, layers):
    """
    Constructs the model
    """

    super(MyModel, self).__init__()

    self.layers_ = layers


  def call(self, inputs):
    """
    Performs a forward step
      Args:
        inputs: our preprocessed input data, we send through our model
      Results:
        output: the predicted output of our input data
    """

    output = inputs
    for layer in self.layers_:
        output = layer(output)
    
    return output