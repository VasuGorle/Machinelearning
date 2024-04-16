import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, LSTM, Reshape, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

# Initialize image data generator with rescaling
train_data_gen = ImageDataGenerator(rescale=1./255)
validation_data_gen = ImageDataGenerator(rescale=1./255)

# For the training generator
train_generator = train_data_gen.flow_from_directory(
    'C:\\Users\\vasug\PycharmProjects\pythonProject2\images\\train',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical'
)

# For the validation generator
validation_generator = validation_data_gen.flow_from_directory(
    'C:\\Users\\vasug\PycharmProjects\pythonProject2\images\\test',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical'
)

# Assuming train_labels and validation_labels are your raw labels
train_labels = to_categorical(train_generator.classes, num_classes=7)
validation_labels = to_categorical(validation_generator.classes, num_classes=7)

# create model structure
emotion_model = Sequential()

emotion_model.add(Conv2D(64, (3, 3), padding='same', input_shape=(48, 48, 1)))
#emotion_model.add(BatchNormalization())
emotion_model.add(Activation('relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, (5, 5), padding='same'))
#emotion_model.add(BatchNormalization())
emotion_model.add(Activation('relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(512, (3, 3), padding='same'))
#emotion_model.add(BatchNormalization())
emotion_model.add(Activation('relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(512, (3, 3), padding='same'))
#emotion_model.add(BatchNormalization())
emotion_model.add(Activation('relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Reshape((1, -1)))

emotion_model.add(LSTM(128, return_sequences=True))
emotion_model.add(LSTM(64))

emotion_model.add(Dense(256))
#emotion_model.add(BatchNormalization())
emotion_model.add(Activation('relu'))
emotion_model.add(Dropout(0.25))

emotion_model.add(Dense(512))
#emotion_model.add(BatchNormalization())
emotion_model.add(Activation('relu'))
emotion_model.add(Dropout(0.25))

emotion_model.add(Dense(7, activation='softmax'))

# Compile the model
emotion_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

# Model callbacks
checkpoint = ModelCheckpoint("./model.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
reduce_learningrate = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, min_delta=0.0001)

callbacks_list = [checkpoint, reduce_learningrate]

# Train the model
emotion_model_info = emotion_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=callbacks_list
    
)

