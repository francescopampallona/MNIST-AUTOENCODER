import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

def prepare_image(data):
  data = data.astype("float32")/255
  
  data = data.reshape(1, np.prod(data.shape[0:]))
  return data

#__________________________________________________________
#Caricamento e preparazione dei dati
(_,_),(data,_) = mnist.load_data()

#__________________________________________________________
#Caricamento dei modelli
encoder = tf.keras.models.load_model("encoder.h5")
decoder = tf.keras.models.load_model("decoder.h5")
#__________________________________________________________
#RAPPRESENTAZIONE DELL'IMMAGINE DA CODIFICARE E DECODIFICARE
INDEX = 12
img = data[INDEX]
plt.imshow(img )
plt.gray()
plt.show()
#__________________________________________________________
#Codifica e decodifica
prepared_img = prepare_image(img)
cod=encoder.predict(prepared_img)
dec= decoder.predict(cod)

#__________________________________________________________
#RAPPRESENTAZIONE DELL'IMMAGINE DECODIFICATA
plt.imshow(dec.reshape(28,28))
plt.gray()
plt.show()