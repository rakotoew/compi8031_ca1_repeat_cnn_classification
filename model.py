from keras import Sequential
from keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator
from data_exploration import CLASSES

IMAGE_SHAPE = (100, 100, 1)
TARGET_SIZE = (100, 100)
BATCH_SIZE = 16


def define_model():
    model = Sequential()
    model.add(Conv2D(60, (5, 5), input_shape=IMAGE_SHAPE, activation='relu'))
    model.add(Conv2D(60, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(CLASSES), activation='softmax'))
    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model(model, training_set, validation_set):
    # Creating dataset iterators
    data_gen = ImageDataGenerator()
    print("Loading training set")
    train_gen = data_gen.flow_from_directory(
        directory=training_set,
        target_size=TARGET_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )
    print("Loading validation set")
    val_gen = data_gen.flow_from_directory(
        directory=validation_set,
        target_size=TARGET_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    # Training model
    return model.fit(train_gen, validation_data=val_gen, epochs=10, validation_steps=8)


def testing_model(model, testing_set):
    print("Loading testing set")
    data_gen = ImageDataGenerator()
    test_gen = data_gen.flow_from_directory(
        directory=testing_set,
        target_size=TARGET_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )
    test_loss, test_acc = model.evaluate(test_gen)
    print('\nTest accuracy:', test_acc)
    print('\nTest loss:', test_loss)
