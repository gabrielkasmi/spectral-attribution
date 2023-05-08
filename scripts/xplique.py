import tensorflow as tf
import os


os.system('pip install xplique')
os.system("pip install pandas==1.4.4")
#os.system('pip install scipy==1.7')
# os.system('pip install opencv-python')

#import cv2
# from waveletsobol_tf.tf_explainer import WaveletSobol

import xplique


#from xplique.attributions import SobolAttributionMethod, Rise, Lime, KernelShap # black box methods
#from xplique.attributions import Saliency, GradientInput, IntegratedGradients, GradCAM, GradCAMPP, VarGrad# white box methods
#from xplique.metrics import Deletion, Insertion, MuFidelity # evaluation methods

#from spectral_sobol.tf_explainer import WaveletSobol
from tensorflow.keras import applications as app # models
# import pandas as pd
import json
import numpy as np
import pandas as pd

print("TensorFlow version:", tf.__version__)


# evaluation of the sobol attributuion method on the xplique benchmark

# load the data
# load the labels and the images

source_dir = '../../data/IN-export'
target_dir = '../../data/spectral-attribution-outputs'

# number of iterations to compute the scores
reps = 1
nb_design = 2
grid_size = 4
batch_size = 128

# labels
labels = pd.read_csv(os.path.join(source_dir, 'labels.csv'))
y_wavelet = labels['label'].values
y = np.array([tf.keras.utils.to_categorical(label, 1000) for label in y_wavelet])

# images
# images = [cv2.imread(os.path.join(source_dir, img_name))[..., ::-1] for img_name in labels['name'].values]

print('Images loaded. Starting the evaluation of the models.')


# helper that will update the file if it exists already
def update(target_dir, filename, data):
  """
  updates the file if it exists, creates and save otherwise
  """

  exists = os.path.exists(os.path.join(target_dir, filename))

  if exists:
    old = json.load(open(os.path.join(target_dir, filename)))
    
    # add the items
    for item in data.keys():
      old[item] = data[item]

    # save the file
    with open(os.path.join(target_dir, filename), 'w') as f:
      json.dump(old, f)

  else:
    with open(os.path.join(target_dir, filename), 'w') as f:
      json.dump(data, f)

##
models = [
    #app.VGG16(classifier_activation = 'softmax'), 
    #app.ResNet50(classifier_activation = 'softmax'),
    #app.MobileNet(classifier_activation = 'softmax'),
    app.efficientnet.EfficientNetB0(classifier_activation = 'softmax'),
]

# preprocessings for each model
preprocessings = [
    app.vgg16.preprocess_input,
    app.resnet.preprocess_input,
    app.mobilenet.preprocess_input,
    app.efficientnet.preprocess_input
]

names = ['VGG', 'ResNet', 'MobileNet', 'EfficientNet']

print(names)

print('essai import xplit')

insertion = xplique.metrics.Insertion(model, x, y)

scores = {}

for name, model, preprocessing in zip(names, models, preprocessings):
  
  x = np.zeros((224,224, 1000))
  insertion = xplique.metrics.Insertion(model, x, y)

  x = preprocessing(np.array(images, copy=True))

  insertion = Insertion(model, x, y)
  deletion = Deletion(model, x, y)
  # fidelity = MuFidelity(model, x, y, batch_size, nb_samples=25)

  print(name)

  # explain the logits rather than the softmax
  model.layers[-1].activation = tf.keras.activations.softmax # 
  

  # set up the explainer and compute the explanations
  wavelet = WaveletSobol(model, grid_size = grid_size, nb_design = nb_design, batch_size = batch_size)
  explanations = wavelet(x,y_wavelet)

  # record the scores
  scores[name] = {
      'insertion' : insertion(wavelet.spatial_cam),
      'deletion'   : deletion(wavelet.spatial_cam)
  }

  # save the results
  update(target_dir, "accuracy.json", scores)
  print("Model {} saved.".format(name))