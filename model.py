# Importing necessary libraries and modules for the project.
import seaborn as sns
sns.set_theme(style="whitegrid")
import matplotlib.pyplot as plt
import datetime
import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from keras import regularizers
from keras.layers import Conv2D, Dense, BatchNormalization, Dropout, MaxPooling2D, Flatten
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator, load_img

# Defining the variables for the train & test directory
train_directory = './train/'
test_directory = './test/'

# Defining the dimensions of the input image
image_row = 48
image_col = 48

# Calculating the number of classes in train_directory
num_classes = len(os.listdir(train_directory))
print(num_classes)


# For the training set, it iterates over the subdirectories in the train_directory. For each subdirectory, 
# it prints the name of the folder and the count of images found within that folder.
print("Train Set:")
train_counts = []
for folder in os.listdir(train_directory):
    print(folder, "folder contains\t\t", len(os.listdir(train_directory + folder)), "image")
    train_counts.append(len(os.listdir(train_directory + folder)))
print()


print("Test Set:")
test_counts = []
for folder in os.listdir(test_directory):
    print(folder, "folder contains\t\t", len(os.listdir(test_directory + folder)), "images")
    test_counts.append(len(os.listdir(test_directory + folder)))


# This generates bar plots to visualize the distribution of images across different classes. 
# The bar plots show the class names on the y-axis and the corresponding image counts on the x-axis. 
plt.figure(figsize=(8, 4))
ax = sns.barplot(y=os.listdir(train_directory),
                 x=train_counts,
                 orientation="horizontal",
                 ).set(title='Train Classes')
plt.show()
print()

ax = sns.barplot(y=os.listdir(test_directory),
                 x=test_counts,
                 orientation="horizontal",
                 ).set(title='Test Classes')
plt.show()
print()

plt.figure(figsize=(20, 20))

# Displaying a grid of sample images from each class in the training dataset. 
# It creates a figure with a size of 20x20 inches and uses a subplot to arrange the images in a grid layout. 
i = 1
for folder in os.listdir(train_directory):
    img = load_img((train_directory + folder + '/' + os.listdir(train_directory + folder)[1]))
    plt.subplot(1, 7, i)
    plt.imshow(img)
    plt.title(folder)
    plt.axis('off')
    i += 1
plt.show()

# The ImageDataGenerator is configured with various data augmentation techniques such as rescaling, zooming, & 
# horizontal flipping. It will generate batches of training data by reading images from the specified directory.
train_data_generator = ImageDataGenerator(rescale=1. / 255,
                                          zoom_range=0.3,
                                          horizontal_flip=True)

training_set = train_data_generator.flow_from_directory(train_directory,
                                                        batch_size=64,
                                                        target_size=(48, 48),
                                                        shuffle=True,
                                                        color_mode='grayscale',
                                                        class_mode='categorical')


test_data_generator = ImageDataGenerator(rescale=1. / 255)

test_set = test_data_generator.flow_from_directory(test_directory,
                                                   batch_size=64,
                                                   target_size=(48, 48),
                                                   shuffle=True,
                                                   color_mode='grayscale',
                                                   class_mode='categorical')

print(training_set.class_indices)

# Building a custom convolutional neural network (CNN) model. The model architecture consists of multiple layers
def create_custom_model(input_size, classes=7):
    model = tf.keras.models.Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_size))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.25))

    model.add(
        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(classes, activation='softmax'))

    model.compile(optimizer=Adam(lr=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


custom_model = create_custom_model((image_row, image_col, 1), num_classes)
print(custom_model.summary())

# Here we set up various callbacks to be used during the training of the model.
checkpoint_path = 'custom_model.h5'
log_directory = "checkpoint/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                             save_best_only=True,
                             verbose=1,
                             mode='min',
                             monitor='val_accuracy')

early_stop = EarlyStopping(monitor='val_accuracy',
                           min_delta=0,
                           patience=3,
                           verbose=1,
                           restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_accuracy',
                              factor=0.2,
                              patience=6,
                              verbose=1,
                              min_delta=0.0001)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_directory, histogram_freq=1)
csv_logger = CSVLogger('training.log')

callbacks = [checkpoint, reduce_lr, csv_logger]

# Finally, fitting the model and performance evaluation
steps_per_epoch = training_set.n // training_set.batch_size
validation_steps = test_set.n // test_set.batch_size

history = custom_model.fit(x=training_set,
                           validation_data=test_set,
                           epochs=150,
                           callbacks=callbacks,
                           steps_per_epoch=steps_per_epoch,
                           validation_steps=validation_steps)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

train_loss, train_accuracy = custom_model.evaluate(training_set)
test_loss, test_accuracy = custom_model.evaluate(test_set)
print("Final train accuracy = {:.2f}, validation accuracy = {:.2f}".format(train_accuracy * 100, test_accuracy * 100))
