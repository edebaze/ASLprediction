import tensorflow as tf


class Callback(tf.keras.callbacks.Callback):
    def __init__(self, callback_params):
        self.loss_threshold = callback_params['loss_threshold']
        self.accuracy_threshold = callback_params['accuracy_threshold']

    # __________________________________________________________________________________________________________________
    # LOSS_THRESHOLD Modifiers
    @property
    def loss_threshold(self):
        return self.__loss_threshold

    @loss_threshold.setter
    def loss_threshold(self, loss_threshold: float) -> None:
        """
            Check that self.__loss_threshold is set with a float between 0 and 1
        :param loss_threshold: loss threshold value to stop the model training
        """
        if loss_threshold < 0:
            print("loss_threshold must be >= 0 - given {:.2f}%".format(loss_threshold))
        if loss_threshold > 1:
            print("loss_threshold must be <= 1 - given {:.2f}%".format(loss_threshold))

        self.__loss_threshold = loss_threshold

    # __________________________________________________________________________________________________________________
    # ACCURACY_THRESHOLD Modifiers
    @property
    def accuracy_threshold(self):
        return self.__accuracy_threshold

    @accuracy_threshold.setter
    def accuracy_threshold(self, accuracy_threshold: float) -> None:
        """
            Check that self.__accuracy_threshold is set with a float between 0 and 1
        :param accuracy_threshold: accuracy threshold value to stop the model training
        """
        if accuracy_threshold < 0:
            print("accuracy_threshold must be >= 0 - given {:.2f}%".format(accuracy_threshold))
        if accuracy_threshold > 1:
            print("accuracy_threshold must be <= 1 - given {:.2f}%".format(accuracy_threshold))

        self.__accuracy_threshold = accuracy_threshold

    # __________________________________________________________________________________________________________________
    def on_epoch_end(self, epoch: int, logs={}):
        if logs.get('val_loss') <= self.LOSS_THRESHOLD and logs.get('val_acc') >= self.ACCURACY_THRESHOLD:
            print("\nReached", self.ACCURACY_THRESHOLD * 100, "accuracy, Stopping!")
            self.model.stop_training = True
