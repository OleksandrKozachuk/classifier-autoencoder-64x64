import os
import sys

import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from utils import CalculateConfusionMatrix


def get_parser():
    from argparse import ArgumentParser

    parser = ArgumentParser(description='classification for datset, and evaluate the trained model')
    parser.add_argument('--model_path', help='full path to trained keras model',
                        default='models/default_images/cnn/cnn.h5')
    parser.add_argument('--data_dir', help='full path to test images, expects a folder with sub-folder for each class',
                        default='data/')

    return parser


def main():
    data_dir = args.data_dir
    model_path = args.model_path
    cm_img_path = os.path.splitext(model_path)[0] + "_cm.jpg"

    target_names = os.listdir(data_dir)
    model = load_model(model_path)
    #
    pred_list = []
    true_name_list = []
    pred_name_list = []
    for class_dir in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_dir)
        if os.path.isdir(class_path):
            class_name = str(class_dir)
            for files in os.scandir(class_path):
                if files.is_file() and (files.name.endswith('.png') or files.name.endswith('.jpg')):

                    image_pth = os.path.join(class_path, files.name)
                    test_image_in = image.load_img(image_pth)
                    test_image = image.img_to_array(test_image_in)
                    test_image = np.expand_dims(test_image, axis=0) * 1. / 255

                    preds = model.predict(test_image)
                    id_pred = np.argmax(preds)
                    if target_names[id_pred] == class_name:
                        pred_list.append(1)
                    else:
                        pred_list.append(0)
                    true_name_list.append(class_name)
                    pred_name_list.append(target_names[id_pred])

    accuracy = pred_list.count(1) / len(pred_list)
    CalculateConfusionMatrix(true_name_list, pred_name_list, target_names, cm_img_path, accuracy)


if __name__ == "__main__":
    # parsing args
    args = get_parser().parse_args(args=None if sys.argv[1:] else ['--help'])
    # run application
    main()
