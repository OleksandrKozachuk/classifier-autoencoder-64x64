import os
import sys

import tensorflow
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler

from networks import *
from utils import *

gpus = tensorflow.config.list_physical_devices('GPU')
tensorflow.config.experimental.set_memory_growth(gpus[0], True)


# generate same image patch for input and output
def fixed_generator(generator):
    for batch in generator:
        yield batch, batch


def get_parser():
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Train Autoencoder network')
    parser.add_argument('--data', help='data directory', default='./data/')
    parser.add_argument('--save_model_path', help='full path to save the trained model', default='models/autoencoder')
    parser.add_argument('--img_size', help='input image size', type=int, default=64)
    parser.add_argument('--batch_size', help='batch size for training', type=int, default=32)
    parser.add_argument('--num_epochs', help='number of epochs for training', type=int, default=50)

    return parser


def main():
    img_width = img_height = args.img_size
    batch_size = args.batch_size
    data_dir = args.data
    epochs = args.num_epochs
    out_dir = args.save_model_path

    save_model_name = out_dir.split('/')[-1] + ".h5"
    save_model_path = os.path.join(out_dir, save_model_name)
    out_dir_imgs = os.path.join(out_dir, "imgs")
    if not os.path.exists(out_dir_imgs):
        os.makedirs(out_dir_imgs)

    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       vertical_flip=True,
                                       horizontal_flip=True,
                                       validation_split=0.2)  # val 20%

    val_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

    train_data = train_datagen.flow_from_directory(data_dir,
                                                   target_size=(img_width, img_height),
                                                   color_mode='rgb',
                                                   batch_size=batch_size,
                                                   class_mode=None,
                                                   shuffle=True,
                                                   subset='training')

    val_data = val_datagen.flow_from_directory(data_dir,
                                               target_size=(img_width, img_height),
                                               color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode=None,
                                               shuffle=False,
                                               subset='validation')

    input_img = Input(shape=(img_width, img_height, 3))
    encoded = Net_Encoder(input_img)
    decoded = Net_Decoder(encoded)
    autoencoder_cnn = Model(input_img, decoded)
    autoencoder_cnn.summary()

    autoencoder_cnn.compile(optimizer=Adam(learning_rate=0.00015), loss='binary_crossentropy')

    # Save autoencoder input and output each epoch
    save_out_imgs = SaveOutputImages(val_data, out_dir_imgs)

    history = autoencoder_cnn.fit(
        fixed_generator(train_data),
        steps_per_epoch=train_data.samples // batch_size,
        epochs=epochs,
        validation_data=fixed_generator(val_data),
        validation_steps=val_data.samples // batch_size,
        callbacks=[save_out_imgs, LearningRateScheduler(lr_schedule_ae)])

    autoencoder_cnn.save(save_model_path)

    PlotTrainValLoss(history, out_dir, epochs)

    # Test random images and visualize the results
    x_test = val_data.next()
    decoded_imgs = autoencoder_cnn.predict(x_test)
    VisualizeAE(x_test, decoded_imgs, out_dir, epochs)


if __name__ == "__main__":
    # parsing args
    args = get_parser().parse_args(args=None if sys.argv[1:] else ['--help'])
    # run application
    main()
