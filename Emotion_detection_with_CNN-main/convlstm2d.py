import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Reshape,Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# Initialize image data generator with rescaling and validation split
train_data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Use the same directory for both training and validation
train_generator = train_data_gen.flow_from_directory(
    "C:\\Users\\vasug\Downloads\images1\images\\train",
    target_size=(68, 68),
    batch_size=48,
    color_mode="grayscale",
    class_mode='categorical',
    subset='training')  # Specify training subset

validation_generator = train_data_gen.flow_from_directory(
    "C:\\Users\\vasug\Downloads\images1\images\\train",
    target_size=(68, 68),
    batch_size=48,
    color_mode="grayscale",
    class_mode='categorical',
    subset='validation')  # Specify validation subset

# Assuming 'model' is a Sequential model
model = Sequential()

# CNN layers
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(68, 68, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Reshape((1, -1)))

model.add(LSTM(64))

model.add(Dense(1028, activation='relu'))
model.add(Dense(3, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])

model.summary()

model_info = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.n // train_generator.batch_size,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=validation_generator.n // validation_generator.batch_size)
