import cv2 as cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras 


labels = ['character_10_yna','ट ','character_12_thaa','ड','ध','ण','त','थ','द','ध','क','न','प','फ','ब',
          'भ','म','य','र','ल','व','ख','श','ष','स','ह','क्ष','त्र','ज्ञ','ग','घ','character_5_kna','च','छ',
          'ज','झ','०','१','२','३','४','५','६','७','८','९']
          
model = tf.keras.models.load_model('AksharCnnNew')

def predict(image):
    resized_image = cv2.resize(np.array(image), (32, 32))
    grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    grayscale_image = np.invert(grayscale_image) / 255
    grayscale_image = np.array([grayscale_image])
    pred = model.predict(grayscale_image)
    return labels[np.argmax(pred)]


