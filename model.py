from keras import Sequential
from keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator
from data_exploration import CLASSES

IMAGE_SHAPE = (100, 100, 1)
TARGET_SIZE = (100, 100)
BATCH_SIZE = 64


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
    return model.fit(train_gen, validation_data=val_gen, epochs=20, validation_steps=10)


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


def test_prediction(model):
    # trying to make prediction from image not from the dataset
    img1 = io.imread(
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.mm.bing.net%2Fth%3Fid%3DOIP.HpaO_qT7muIzxU-cmzh64AHaEK%26pid%3DApi&f=1")
    img2 = io.imread(
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.MoEPeScC5DFakg1wvZXt6gHaE7%26pid%3DApi&f=1")
    img3 = io.imread(
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.q8BUZJA88aMSYpdbJS__MAHaEL%26pid%3DApi&f=1")
    img4 = io.imread(
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.2kYdR51cGD8rFh5gae-_jAHaFR%26pid%3DApi&f=1")
    img5 = io.imread(
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse2.mm.bing.net%2Fth%3Fid%3DOIP.Ri6v7qBWzgDoVnVnaEJ0QwHaEo%26pid%3DApi&f=1")
    img6 = io.imread(
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse2.mm.bing.net%2Fth%3Fid%3DOIP.r3YEBhjTVEVWFIchuGLbfgHaFj%26pid%3DApi&f=1")
    img7 = io.imread(
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.YUf2wOBWOv0mTvecLHyAoAHaEA%26pid%3DApi&f=1")
    img8 = io.imread(
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.h2PxvwAkY8Vt_loZW8WHMgHaEK%26pid%3DApi&f=1")
    img9 = io.imread(
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.mm.bing.net%2Fth%3Fid%3DOIP.8FTi2C-pI04OVXhdhGA7twHaE7%26pid%3DApi&f=1")
    images = [img1, img2, img3, img4, img5, img6, img7, img8, img9]
    for image in images:
        # showing the image
        plt.imshow(image)

        # pre processing the image
        image = utils.resize_image(image, 100)
        image = utils.gray_scale(image)
        image.reshape(100, 100, 1)
        predict = CLASSES[np.argmax(model.predict(image))]
        print(predict)

