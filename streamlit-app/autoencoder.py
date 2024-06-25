import tensorflow as tf
from tensorflow.keras import regularizers, layers, losses
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import Callback
import streamlit as st
import time
import plotly.express as px
import plotly.graph_objects as go

class StopTrainingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if st.session_state.get('stop_training', False):
            self.model.stop_training = True
            message = st.empty()
            message.warning("Oprire construire model...")
            time.sleep(2)  
            message.empty()  
            st.session_state['stop_training'] = False 

# Define the Autoencoder class
class Autoencoder(Model):
    def __init__(self, input_shape):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(16, activation='relu'),
            layers.Dense(8, activation='relu'),
            layers.Dense(4, activation='relu')
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(8, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(input_shape, activation='linear')
        ])
    
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
class ProgressCallback(Callback):
    def __init__(self, st_progress_bar):
        super().__init__()
        self.st_progress_bar = st_progress_bar

    def on_epoch_end(self, epoch, logs=None):
        self.st_progress_bar.progress((epoch + 1) / 100)

class LossHistory(Callback):
    def __init__(self):
        super(LossHistory, self).__init__()
        self.train_losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.train_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
