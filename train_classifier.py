import os
import sys

from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler

from networks import *
from utils import *


def get_parser():
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Train models pretrained with AutoEncoders for classification')
    parser.add_argument('--data', help='data directory', default='./data/')
    parser.add_argument('--save_model_path', help='full path to save the trained model', default='models/cnn/')
    parser.add_argument('--base_model_path', help='full path to input autoencoder model (.h5)',
                        default='models/autoencoder/autoencoder.h5')

    parser.add_argument('--img_size', help='input image size', type=int, default=64)
    parser.add_argument('--number_of_classes', help='number of classes in the dataset', type=int)
    parser.add_argument('--num_epochs', help='number of epochs for training', type=int, default=100)
    parser.add_argument('--batch_size', help='batch size for training', type=int, default=32)
    parser.add_argument('--freeze_feature_extractor', action='store_true')  # default is false
    parser.add_argument('--train_from_scratch', action='store_true')  # default is false
    parser.add_argument('--extra_dense_layer', action='store_true')  # default is false

    return parser


def main():
    data_dir = args.data
    out_dir = args.save_model_path
    base_model_path = args.base_model_path
    freeze = args.freeze_feature_extractor
    train_from_scratch = args.train_from_scratch
    epochs = args.num_epochs
    img_width = img_height = args.img_size
    num_classes = args.number_of_classes
    batch_size = args.batch_size
    add_extra_dense = args.extra_dense_layer

    save_model_name = out_dir.split('/')[-1] + "cnn.h5"
    save_model_path = os.path.join(out_dir, save_model_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if train_from_scratch:
        print("Training the classification model from scratch ...")
        input_img = Input(shape=(img_width, img_height, 3))
        net = Net_Encoder(input_img)
        if add_extra_dense:
            net = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0001), name='cls_dense')(net)
        predictions = Dense(num_classes, activation='softmax')(net)
        model = Model(inputs=input_img, outputs=predictions)
    else:
        orig_model = load_model(base_model_path)  # create the base pre-trained model
        base_model = Model(inputs=orig_model.input, outputs=orig_model.get_layer('latent_feats').output)
        print('Pretrained model loaded ....')
        net = base_model.output
        # if add_extra_dense:
        #     x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0001), name='cls_dense')(x)
        predictions = Dense(num_classes, activation='softmax')(net)
        model = Model(inputs=base_model.input, outputs=predictions)
        if freeze:
            # train only the top layers (which were randomly initialized), i.e. freeze all convolutional layers
            print("all feature extractor freezed ... ")
            for layer in base_model.layers:
                layer.trainable = False
    model.summary()

    optimizer = SGD(lr=0.001, momentum=0.9, decay=1e-6)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # this is the augmentation we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        featurewise_center=True,
        featurewise_std_normalization=True,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=90,
        width_shift_range=0.1,
        height_shift_range=0.1,
        validation_split=0.2)
    #
    # train_datagen.mean = GetCifar10Mean()
    # train_datagen.std = GetCifar10STD()

    test_datagen = ImageDataGenerator(rescale=1. / 255,
                                      featurewise_center=True,
                                      featurewise_std_normalization=True,
                                      validation_split=0.2)

    # test_datagen.mean = GetCifar10Mean()
    # test_datagen.std = GetCifar10STD()

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_width, img_height),
        color_mode='rgb',
        batch_size=batch_size,
        class_mode='categorical',
        subset='training')

    validation_generator = test_datagen.flow_from_directory(
        data_dir,
        target_size=(img_width, img_height),
        color_mode='rgb',
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation')

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        shuffle=True,
        workers=8,
        verbose=1,
        callbacks=[LearningRateScheduler(lr_schedule)])

    model.save(save_model_path)

    PlotTrainValAccuracy(history, out_dir, epochs)
    PlotTrainValLoss(history, out_dir, epochs)


if __name__ == "__main__":
    # parsing args
    args = get_parser().parse_args(args=None if sys.argv[1:] else ['--help'])
    # run application
    main()
