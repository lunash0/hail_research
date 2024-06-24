import os
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Conv2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.losses import categorical_crossentropy, binary_crossentropy
from tensorflow.keras import callbacks
from tensorflow.keras.metrics import top_k_categorical_accuracy


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


dataset_root_dir = "/home/hail09/Documents/hail_moon/CIFAR-100_classification&object_detection/cifrar-100"
train_images_dir = os.path.join(dataset_root_dir, 'train_image')
test_images_dir = os.path.join(dataset_root_dir, 'test_image')
meta_path = os.path.join(dataset_root_dir, 'meta')
train_path = os.path.join(dataset_root_dir, 'train')
test_path = os.path.join(dataset_root_dir, 'test')

meta_data = unpickle(meta_path)
data_train_dict = unpickle(train_path)  # idx: b'filenames', b'batch_label', b'fine_labels', b'coarse_labels', b'data'
data_test_dict = unpickle(test_path)  # idx: b'filenames', b'batch_label', b'fine_labels', b'coarse_labels', b'data'

superclass_dict = dict(list(enumerate(meta_data[b'coarse_label_names'])))

X_train = data_train_dict[b'data']
y_train = np.array(data_train_dict[b'coarse_labels'])

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.8, stratify=y_train)
X_train = X_train.reshape(len(X_train), 3, 32, 32).transpose(0, 2, 3, 1)  # (batch_size, height, width, channels)
X_valid = X_valid.reshape(len(X_valid), 3, 32, 32).transpose(0, 2, 3, 1)
X_train_1 = X_train / 255.
X_valid_1 = X_valid / 255.

X_test = data_test_dict[b'data']
X_test = X_test.reshape(len(X_test), 3, 32, 32).transpose(0, 2, 3, 1)
y_test = np.array(data_test_dict[b'coarse_labels'])

y_train = to_categorical(y_train, 100)
y_test = to_categorical(y_test, 100)
y_valid = to_categorical(y_valid, 100)

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

X_train_augmented = []
y_train_augmented = []

for class_index in range(20):
    X_class = X_train[y_train[:, class_index] == 1]
    y_class = y_train[y_train[:, class_index] == 1]
    for X_sample, y_sample in zip(X_class, y_class):
        for X_augmented, y_augmented in datagen.flow(X_sample.reshape(1, 32, 32, 3), y_sample.reshape(1, 100),
                                                     batch_size=1):
            X_train_augmented.append(X_augmented[0])
            y_train_augmented.append(y_augmented[0])
            break

X_train_augmented = np.array(X_train_augmented)
y_train_augmented = np.array(y_train_augmented)

X_train = np.concatenate([X_train, X_train_augmented])
y_train = np.concatenate([y_train, y_train_augmented])

X_train, y_train = shuffle(X_train, y_train)

INPUT_SHAPE = (32, 32, 3)

base_model = ResNet50(input_shape=INPUT_SHAPE,
                      include_top=False,
                      weights='imagenet')

base_model.summary()
base_model.trainable = True
set_trainable = False

for layer in tqdm(base_model.layers):
    if layer.name in ['res5a_branch2a', 'res5a_branch2b', 'res5a_branch2c',
                      'res5b_branch2a', 'res5b_branch2b', 'res5b_branch2c',
                      'res5c_branch2a', 'res5c_branch2b', 'res5c_branch2c']:
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

layers = [(layer, layer.name, layer.trainable) for layer in base_model.layers]
# layers_df = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
# print(layers_df)


model = Sequential()

model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(len(superclass_dict), activation='softmax'))

model.compile(loss=categorical_crossentropy,
              optimizer=Adam(learning_rate=0.0001),
              metrics=['acc', 'top_k_categorical_accuracy'])

model.summary()

EPOCHS = 200
BATCH_SIZE = 64

early_stopping = callbacks.EarlyStopping(monitor="val_loss",
                                         patience=20)  # if val loss decrease for 5 epochs in a row, stop training

# Create ModelCheckpoint callback to save best model during fine-tuning
# checkpoint_path = "model_checkpoints/"
# model_checkpoint = callbacks.ModelCheckpoint(checkpoint_path,
#                                              save_best_only=True,
#                                              monitor="val_loss")

# Creating learning rate reduce callback
reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss",
                                        factor=0.2,  # multiply the learning rate by 0.2 (reduce by 5x)
                                        patience=10,
                                        verbose=1,
                                        min_lr=1e-7)

history = model.fit(X_train,
                    y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    use_multiprocessing=True,
                    validation_data=(X_valid, y_valid),
                    callbacks=[early_stopping,  # stop model after X epochs of no improvements
                               reduce_lr]  # + model_checkpoint,  # save only the best model during training
                    )

# def plot_history(history):
#     val_loss = history.history['val_loss' ]
#     loss =     history.history['loss' ]
#     acc =      history.history['accuracy' ]
#     val_acc =  history.history['val_accuracy' ]
#
#     epochs    = range(1,len(acc)+1,1)
#
#     plt.plot  ( epochs,     acc, 'r--', label='Training acc'  )
#     plt.plot  ( epochs, val_acc,  'b', label='Validation acc')
#     plt.title ('Training and validation accuracy')
#     plt.ylabel('acc')
#     plt.xlabel('epochs')
#     plt.legend()
#
#     plt.figure()
#
#     plt.plot  ( epochs,     loss, 'r--', label='Training loss' )
#     plt.plot  ( epochs, val_loss ,  'b', label='Validation loss' )
#     plt.title ('Training and validation loss')
#     plt.ylabel('loss')
#     plt.xlabel('epochs')
#     plt.legend()
#     plt.figure()
#
# plot_history(history)

model_evaluate = model.evaluate(X_valid)
