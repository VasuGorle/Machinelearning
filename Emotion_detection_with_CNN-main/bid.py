from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Reshape, LSTM, BatchNormalization,ConvLSTM2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# Initialize image data generator with rescaling
train_data_gen = ImageDataGenerator(rescale=1./255)
validation_data_gen = ImageDataGenerator(rescale=1./255)

train_generator = train_data_gen.flow_from_directory(
        "C:\\Users\\vasug\Downloads\images1\images\\train",
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

validation_generator = validation_data_gen.flow_from_directory(
        "C:\\Users\\vasug\Downloads\images1\images\\validation",
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

model = Sequential()

time_steps, rows, cols, channels = 5, 48, 48, 1

model = Sequential()

model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), input_shape=(time_steps, rows, cols, channels), padding='same', return_sequences=True))
model.add(BatchNormalization())

model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), input_shape=(time_steps, rows, cols, channels), padding='same', return_sequences=True))
model.add(BatchNormalization())
model.add(ConvLSTM2D(filters=128, kernel_size=(3, 3), input_shape=(time_steps, rows, cols, channels), padding='same', return_sequences=True))
model.add(BatchNormalization())
model.add(ConvLSTM2D(filters=128, kernel_size=(3, 3), input_shape=(time_steps, rows, cols, channels), padding='same', return_sequences=True))
model.add(BatchNormalization())
model.add(ConvLSTM2D(filters=256, kernel_size=(3, 3), input_shape=(time_steps, rows, cols, channels), padding='same', return_sequences=True))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Reshape((1, -1)))

# Adding Bidirectional LSTM layer

model.add(Flatten())

# Fully connected layers for classification
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))  # Assuming 5 classes for emotions

# Compile the model (customize as needed)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model_info = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.n // train_generator.batch_size,
        epochs=20,
        validation_data=validation_generator,
        validation_steps=validation_generator.n // validation_generator.batch_size)
