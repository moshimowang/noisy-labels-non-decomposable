import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


## Loss functions

def crossentropy(y_true, y_pred):
    # this gives the same result as using keras.objective.crossentropy
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    return -K.sum(y_true * K.log(y_pred), axis=-1)


def robust(name, P):

    if name == 'backward':
        P_inv = K.constant(np.linalg.inv(P))

        def loss(y_true, y_pred):
            y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
            y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
            return -K.sum(K.dot(y_true, P_inv) * K.log(y_pred), axis=-1)

    elif name == 'forward':
        P = K.constant(P)

        def loss(y_true, y_pred):
            y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
            y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
            return -K.sum(y_true * K.log(K.dot(y_pred, P)), axis=-1)

    return loss


## Models

class KerasModel():

    # custom losses
    def make_loss(self, loss, P=None):

        if loss in ['crossentropy', 'plug', 'est_plug']:
            return crossentropy
        elif loss in ['forward', 'backward']:
            return robust(loss, P)
        else:
            ValueError("Loss unknown.")

    def compile(self, model, loss, P=None):

        if self.optimizer is None:
            ValueError()

        metrics = ['accuracy']

        model.compile(loss=self.make_loss(loss, P), optimizer=self.optimizer, metrics=metrics)

        model.summary()
        
        self.model = model

    def load_model(self, file):
        
        self.model.load_weights(file)
        print('Loaded model from %s' % file)

    def fit_model(self, model_file, X_train, Y_train, validation_split=None,
                  validation_data=None):

        # cannot do both
        if validation_data and validation_split:
            return ValueError()

        callbacks = []
        monitor = 'val_loss'

        mc_callback = tf.keras.callbacks.ModelCheckpoint(model_file, monitor=monitor,
                                      verbose=1, save_best_only=True)
        callbacks.append(mc_callback)

        if hasattr(self, 'scheduler'):
            callbacks.append(self.scheduler)

        history = self.model.fit(
                        X_train, Y_train, batch_size=self.num_batch,
                        epochs=self.epochs,
                        validation_split=validation_split,
                        validation_data=validation_data,
                        verbose=1, callbacks=callbacks)

        # use the model that reached the lowest loss at training time
        self.load_model(model_file)

        return history.history

    def predict_proba(self, X):
        pred = self.model.predict(X, batch_size=self.num_batch, verbose=1)
        return pred
    
    def predict(self, X):
        pred = self.model.predict(X, batch_size=self.num_batch, verbose=1)
        return np.argmax(pred, axis=1)
    
    def predict_plug(self, X, noise_T_inv):
        probs = self.predict_proba(X)
        corrected_probs = np.dot(probs, noise_T_inv.T)
        return np.argmax(corrected_probs, axis=1)


class UCIModel(KerasModel):

    def __init__(self, features, classes, num_batch=32, epochs=50):
        self.features = features
        self.classes = classes
        self.num_batch = num_batch
        self.epochs = epochs
        self.optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.01)

    def build_model(self, loss, P=None):
        # make sure P is row stochastic
        np.testing.assert_allclose(np.sum(P, axis=1), 1)

        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(self.classes, input_shape=(self.features,)),
            tf.keras.layers.Softmax()
            ])

        self.compile(model, loss, P)
