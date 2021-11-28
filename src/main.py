import tensorflow as tf
import matplotlib.pyplot as plt
from util import visualize, loading_data, test
from model import MyModel
from classify import classify

#Thanks to Hannah-Richert and leonl42 for providing their code from previous tasks.

if __name__ == "__main__":

    # hyperparameters
    epochs = 20
    learning_rate = 0.05

    # lists vof visualization
    results = []
    trained_models = []
    train_losses = []
    valid_losses = []
    valid_accuracies = []

    # getting our data
    train_ds,valid_ds,test_ds = loading_data()

    # training multiple models with different conditions
    for optimizer in [tf.keras.optimizers.SGD(learning_rate),tf.keras.optimizers.Adam(learning_rate)]:
        for model in [MyModel(None,0),MyModel('l1_l2',0),MyModel(None,0.5),MyModel('l1_l2',0.5)]:
            tf.keras.backend.clear_session()
            result,model = classify(model,optimizer,epochs,train_ds,valid_ds)
            results.append(result)
            trained_models.append(model)

    # splitting our results into multiple lists
    for result in results:
        train_losses.append(result[0])
        valid_losses.append(result[1])
        valid_accuracies.append(result[2])

    # after training the models and adjusting our hyperparameters, 
    # testing the final model on our unseen test_ds
    _, test_accuracy = loss,accuracy = test(trained_models[-1],test_ds,tf.keras.losses.BinaryCrossentropy())
    print("The last model (Adam,l1_l2,Dropout-0.5) has a accuracy on our unseen test_ds of ",test_accuracy, ".")

    # visualize the losses and accuracies in a grid_plot
    visualize(train_losses,valid_losses,valid_accuracies)
    plt.show()