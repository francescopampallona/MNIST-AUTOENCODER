import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np


#ENCODER
def ENCODER():
  input_img = Input(shape=(784,))
  encoded = Dense(128, activation='relu')(input_img)
  encoded = Dense(64, activation='relu')(encoded)
  encoded = Dense(32, activation='relu')(encoded)
  encoder = Model(input_img, encoded)
  return encoder

#DECODER
def DECODER():
  encoded_input = Input(shape=(32,))
  decoded = Dense(64, activation='relu')(encoded_input)
  decoded = Dense(128, activation='relu')(decoded)
  decoded = Dense(784, activation = 'sigmoid')(decoded)
  decoder = Model(encoded_input, decoded)
  return decoder 


#DEFINIZIONE DEL MODELLO DA ADDESTRARE
def autoencoder():
  input_img = Input(shape=(784,))
  encoder = ENCODER()
  decoder = DECODER()
  autoencoder=Model(input_img, decoder(encoder(input_img)))
  #COMPILAZIONE DEL MODELLO
  autoencoder.compile(optimizer='RMSprop', loss='MSE')
  return encoder, decoder, autoencoder


#DATASET PER IL TRAINING E IL TESTING
def dataset():
  (x_train, _), (x_test, _) = mnist.load_data()
  #Normalizzazione dei valori nelle immagini tra 0 e 1
  x_train = x_train.astype("float32")/255
  x_test = x_test.astype("float32")/255
  #Appiattimento delle immagini 28x28 a vettore di 784 elementi
  x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
  x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
  return x_train, x_test

#__________________________________________________________
#Caricamento del dataset e del modello
x_train, x_test = dataset()
encoder, decoder, autoencoder = autoencoder()
#Addestramento dell'autoencoder
h=autoencoder.fit(x_train, x_train, epochs = 100, batch_size = 256, shuffle=True, validation_data=(x_test, x_test))
#Salvataggio 
autoencoder.save("autoencoder.h5")
decoder.save("decoder.h5")
encoder.save("encoder.h5")
#_________________________________________________________
#GRAFICO SULL'ANDAMENTO DELL'ADDESTRAMENTO
print(h.history.keys())
#GRAFICO PER L'ANDAMENTO DELLA LOSS
plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.title('PERDITA DEL MODELLO')
plt.ylabel('perdita')
plt.xlabel('epoca')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()