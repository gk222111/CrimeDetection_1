import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

from tensorflow.keras.applications import DenseNet121
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout,MaxPooling2D , Conv2D,Flatten
from tensorflow.keras.models import Sequential

from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import classification_report


from IPython.display import clear_output
import warnings
warnings.filterwarnings('ignore')

from keras.preprocessing.image import ImageDataGenerator
import os
import cv2
import pickle
from tqdm import tqdm
import numpy as np
import random

# Load the training and test data
train_reduced_path = '/Home/Downloads/UCF/Train'
test_reduced_path = '/Home/Downloads/UCF/Test'

# Define the categories and labels
categories_labels = {'Abuse':0, 'Arrest':1, 'Arson':2, 'Assualt':3, 'Burglary':4, 'Explosion':5, 'Fighting':6, 'NormalVideos':7, 'RoadAccidents':8, 'Robbery':9, 'Shooting':10, 'Shoplifting':11, 'Stealing':12, 'Vandalism':13}

def load_data(base_dir, categories_labels):
    data = []

    # Go through each category
    for category, label in categories_labels.items():
        # The path to the category directory
        category_dir = os.path.join(base_dir, category)

        # Make sure the directory exists
        if os.path.isdir(category_dir):
            # Go through each file in the directory
            for filename in tqdm(os.listdir(category_dir), desc=f"Loading {category}"):
                # Make sure the file is an image
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    # The path to the image
                    image_path = os.path.join(category_dir, filename)

                    try:
                        # Load the image
                        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                        # Resize the image
                        image = cv2.resize(image, (50, 50))

                        # Reshape the image to 4D array (ImageDataGenerator requires 4D array)
                        image = image.reshape((1,) + image.shape + (1,))

                        # Add the image and its label to the data
                        data.append([image, label])
                    except Exception as e:
                        print(f"Error loading image {image_path}: {e}")

    return data

# Load the training and test data
training_data = load_data(train_reduced_path, categories_labels)
test_data = load_data(test_reduced_path, categories_labels)

# Combine the training and test data
total_data = training_data + test_data

print(f"Loaded {len(total_data)} images.")

# Load the training and test data
train_reduced_path = '/content/ucf-crime-dataset/Train_reduced'
test_reduced_path = '/content/ucf-crime-dataset/Test_reduced'

crime_types=os.listdir(train_reduced_path)
n=len(crime_types)
print("Number of Crime categories : ",n)


crimes={}
train=test=0
for clss in crime_types:
    num=len(os.listdir(os.path.join(train_reduced_path,clss)))
    train+=num
    test+=len(os.listdir(os.path.join(test_reduced_path,clss)))

    crimes[clss]=num
    
    
    
colors= ['purple', 'brown']
plt.figure(figsize=(10, 5))
plt.pie(x=np.array([train,test]), autopct="%.1f%%", explode=[0.1, 0.1], labels=["Training Data", "Test Data"], pctdistance=0.5, colors=colors)
plt.title("Share of train and test images ", fontsize=8);



plt.figure(figsize=(8,5))
plt.bar(list(crimes.keys()), list(crimes.values()), width=0.5,align="center")
plt.xticks(rotation=90)

plt.xlabel("Types of Reported Crimes")
plt.ylabel("No. of Reported Crimes")
plt.show()



colors = ['purple', 'pink','red', 'green', 'blue', 'orange', 'yellow', 'violet', 'brown']

# Create the pie chart with the specified colors
plt.figure(figsize=(10, 5))
plt.pie(x=np.array(list(crimes.values())), autopct="%.1f%%", explode=[0.1] * n, labels=list(crimes.keys()), pctdistance=0.7, colors=colors)
plt.title("Share of train and test images", fontsize=8)

# Show the plot
plt.show()
print(os.listdir(train_reduced_path))
print(os.listdir(test_reduced_path))


train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest',
    validation_split=0.2) # set validation split




crime_types = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 'Fighting', 'NormalVideos', 'RoadAccidents', 'Robbery', 'Shooting', 'Shoplifting', 'Stealing', 'Vandalism']


batchSize=32


train_generator = train_datagen.flow_from_directory(
    train_reduced_path,
    target_size=(224, 224),
    batch_size=batchSize,
    classes=crime_types,
    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(
    train_reduced_path, # same directory as training data
    target_size=(224, 224),
    batch_size=batchSize,
    classes=crime_types,
    subset='validation') # set as validation data

test_generator = ImageDataGenerator().flow_from_directory(
    test_reduced_path,
    target_size=(224,224),
    classes=crime_types,
    shuffle= False,
    batch_size = batchSize)# set as test data
    
    
    
print ("In train_generator ")
for cls in range(len (train_generator.class_indices)):
    print(crime_types[cls],":\t",list(train_generator.classes).count(cls))
print ("")

print ("In validation_generator ")
for cls in range(len (validation_generator.class_indices)):
    print(crime_types[cls],":\t",list(validation_generator.classes).count(cls))
print ("")

print ("In test_generator ")
for cls in range(len (test_generator.class_indices)):
    print(crime_types[cls],":\t",list(test_generator.classes).count(cls))
    
    
    
#plots images with labels within jupyter notebook
def plots(ims, figsize = (12,12), rows=5, interp=False, titles=None, maxNum = 9):
    if type(ims[0] is np.ndarray):
        ims = np.array(ims).astype(np.uint8)
        if(ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))

    f = plt.figure(figsize=figsize)
    #cols = len(ims) //rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    cols = maxNum // rows if maxNum % 2 == 0 else maxNum//rows + 1
    #for i in range(len(ims)):
    for i in range(maxNum):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=20)
        plt.imshow(ims[i], interpolation = None if interp else 'none')
        
        
        
train_generator.reset()
imgs, labels = train_generator.next()

#print(labels)

labelNames=[]
labelIndices=[np.where(r==1)[0][0] for r in labels]
#print(labelIndices)

for ind in labelIndices:
    for labelName,labelIndex in train_generator.class_indices.items():
        if labelIndex == ind:
            #print (labelName)
            labelNames.append(labelName)

#labels



plots(imgs, rows=4, titles = labelNames, maxNum=20)

n = len(crime_types)

from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import EarlyStopping



#DenseNet121

def transfer_learning():
  base_model = DenseNet121(weights='imagenet',
                                include_top=False,
                                input_shape=(224, 224,3))
  thr=149
  for layers in base_model.layers[:thr]:
        layers.trainable=False

  for layers in base_model.layers[thr:]:
        layers.trainable=True

  return base_model
  
def create_model():
    model=Sequential()

    base_model=transfer_learning()
    model.add(base_model)

    model.add(GlobalAveragePooling2D())

    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(1024, activation="relu"))

    #model.add(Flatten())

    model.add(Dense(len(crime_types),activation="softmax"))

    model.summary()

    return model
    
    
model=create_model()



# Display the success message.
print("Model Created Successfully!")



# Create an Instance of Early Stopping Callback.
early_stopping_callback = EarlyStopping(monitor = 'val_loss', patience = 15, mode = 'min', restore_best_weights = True)

# Plot the structure of the contructed model.
plot_model(model, to_file='DenseNet121_model_structure_plot.png', show_shapes=True, show_layer_names=True)


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

earlystop_cb = EarlyStopping(patience=10, restore_best_weights=True)
reduce_lr_cb = ReduceLROnPlateau(factor=0.5, patience=5, monitor='val_loss', min_lr=0.00001)
checkpoint_cb = ModelCheckpoint('model.h5', save_best_only=True)

callbacks = [earlystop_cb, reduce_lr_cb, checkpoint_cb]


# Compile the model and specify loss function, optimizer and metrics to the model.
model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ["accuracy"])


stepsPerEpoch= (train_generator.samples+ (batchSize-1)) // batchSize
print("stepsPerEpoch: ", stepsPerEpoch)

validationSteps=(validation_generator.samples+ (batchSize-1)) // batchSize
print("validationSteps: ", validationSteps)


#validationSteps=(test_generator.samples+ (batchSize-1)) // batchSize
#print("validationSteps: ", validationSteps)


train_generator.reset()
validation_generator.reset()

# Fit the model
history = model.fit_generator(
    train_generator,
    validation_data = validation_generator,
    epochs = 4,
    steps_per_epoch = stepsPerEpoch,
    validation_steps= validationSteps,
    callbacks=callbacks,
    verbose=1)
    
    
# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'Validation'], loc='upper left')
plt.show()



# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()



