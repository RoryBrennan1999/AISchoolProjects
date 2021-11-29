# project1_18237606.py
# AI Project 1 CE4041
# Dr Colin Flanagan
# Rory Brennan 18237606
# Keras MNIST Classifier Summary:
# Input -> Convolution (32) -> MaxPool -> Dropout (0.2) -> Convolution (64) 
# -> MaxPool -> Dropout (0.2) -> Dense (hidden) -> Dropout (0.2) -> Dense (output)

EPOCHS  = 20      # Training run parameters.  20 epoch run limit.
SPLIT   = 0.2     # 80%/20% train/val split.
SHUFFLE = True    # Shuffle training data before each epoch.
BATCH   = 32      # Minibatch size (note Keras default is 32).
OPT     = 'rmsprop'  # RMSprop optimizer.

import numpy as np
import matplotlib.pyplot as plt


# Initialise the random number generators to help reduce variance
# between runs. 


np.random.seed(1)                # Initialise system RNG.

import tensorflow
tensorflow.random.set_seed(2)    # and the seed of the Tensorflow backend.

print(tensorflow.__version__)    # Should be at least 2.0.


# Import the relevant Keras library modules.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense          # Fully-connected layer
from tensorflow.keras.layers import Conv2D         # 2-d Convolutional layer
from tensorflow.keras.layers import MaxPooling2D   # 2-d Max-pooling layer
from tensorflow.keras.layers import Flatten        # Converts 2-d layer to 1-d layer
from tensorflow.keras.layers import Activation     # Nonlinearities
from tensorflow.keras.layers import Dropout        # Dropout regularization

from tensorflow.keras.utils import to_categorical


# Load up the MNIST dataset.

from tensorflow.keras.datasets import mnist        


# Extract the training images into "training_inputs" and their 
# associated class labels into "training_labels".  
# 
# There are 60000 28 x 28 8-bit greyscale training images and 
# 10000 test images.

(training_inputs, training_labels), (testing_inputs, testing_labels) = mnist.load_data()

print(training_inputs.shape, training_inputs.dtype, testing_inputs.shape, testing_inputs.dtype)


# The inputs to the network need to be normalised 'float32' 
# values, in a tensor of shape (N,28,28,1).  N is the number of 
# images, each one with 28 rows and 28 columns, and one channel.  
# 
# A greyscale image has one channel (normally implicit), an RGB 
# image would have 3. A convolutional net can work with multiple-
# channel input images, but needs the number of channels to be 
# explicitly stated, hence the final 1 in the tensor shape.

training_images = (training_inputs.astype('float32')/255)[:,:,:,np.newaxis]  # Normalised float32 4-tensor.

categorical_training_outputs = to_categorical(training_labels)

testing_images = (testing_inputs.astype('float32')/255)[:,:,:,np.newaxis]

categorical_testing_outputs = to_categorical(testing_labels)

print(training_images.shape,training_images.dtype)
print(testing_images.shape,testing_images.dtype)
print(categorical_training_outputs.shape, training_labels.shape)
print(categorical_testing_outputs.shape, testing_labels.shape)

plt.figure(figsize=(14,4))
for i in range(20):
    plt.subplot(2,10,i+1)
    plt.imshow(training_images[i,:,:,0],cmap='gray')
    plt.title(str(training_labels[i]))
    plt.axis('off')


# Create a Keras model for a net.
# Input -> Convolution (32) -> MaxPool -> Dropout (0.2) -> Convolution (64) 
# -> MaxPool -> Dropout (0.2) -> Dense (hidden) -> Dropout (0.2) -> Dense (output)

model = Sequential([
            Conv2D(32, kernel_size=5, padding='same', input_shape=training_images.shape[1:]),
            Activation('relu'),
            MaxPooling2D(pool_size=(2,2), strides=(2,2)),
            Dropout(0.2),
            Conv2D(64, kernel_size=5, padding='same', input_shape=training_images.shape[1:]),
            Activation('relu'),
            MaxPooling2D(pool_size=(2,2), strides=(2,2)),
            Dropout(0.2),
            Flatten(),
            Dense(52),
            Activation('relu'),
            Dropout(0.2),
            Dense(10),
            Activation('softmax')
        ])


# Print model summary

print("The Keras network model")
model.summary()

# Cross entropy loss and an RMSprop optimizer.

model.compile(loss='categorical_crossentropy', optimizer=OPT, metrics=['accuracy'])


# Generate a model that fits the training set. Early stopping is
# in place so the full 20 epochs will not be seen.


from tensorflow.keras.callbacks import EarlyStopping

stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, 
                     verbose=2, mode='auto',
                     restore_best_weights=True)


history = model.fit(training_images, categorical_training_outputs,
                    epochs=EPOCHS, 
                    batch_size=BATCH, 
                    shuffle=SHUFFLE, 
                    validation_split=SPLIT,
                    verbose=2, 
                    callbacks=[stop])


# Output figures for training and validation loss.

plt.figure('Training and Validation Losses per epoch', figsize=(8,8))

plt.plot(history.history['loss'],label='training') # Training data error per epoch.

plt.plot(history.history['val_loss'],label='validation') # Validation error per ep.

plt.grid(True)

plt.legend()

plt.xlabel('Epoch Number')
plt.ylabel('Loss')

# Output figures for validation and training Accuracy

plt.figure('Training and Validation Accuracy per epoch',figsize = (8,8))

plt.plot(history.history['accuracy'],label = 'training') # Training accuracy per ep.

plt.plot(history.history['val_accuracy'],label = 'validation') # Validation data accuracy per epoch.

plt.grid(True)

plt.legend()

plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.show()


# Test the performance of the network on the 
# completely separate testing set.


print("Performance of network on testing set:")
test_loss,test_acc = model.evaluate(testing_images,categorical_testing_outputs)
print("Accuracy on testing data: {:6.2f}%".format(test_acc*100))
print("Test error (loss):        {:8.4f}".format(test_loss))


print("Performance of network:")
print("Accuracy on training data:   {:6.2f}%".format(history.history['accuracy'][-1]*100))
print("Accuracy on validation data: {:6.2f}%".format(history.history['val_accuracy'][-1]*100))
print("Accuracy on testing data:    {:6.2f}%".format(test_acc*100))


