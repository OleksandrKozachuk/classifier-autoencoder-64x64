### Minimal requirements
```
python >=3.6.0
```

### Installation
```commandline
python3 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Usage
#### To train autoencoder
```commandline
python3 train_autoencoder.py -h 'to see help'

python3 train_autoencoder.py --data 'train data directory' --save_model_path 'full path to save the trained model'
                             --img_size 'input image size' --batch_size 'batch size for training'
                             --num_epochs 'number of epochs for training'
```
#### To train classifier
```commandline
python3 train_classifier.py -h 'to see help'

python3 train_classifier.py --data 'train data directory' --save_model_path 'full path to save the trained model'
                            --base_model_path 'full path to input autoencoder model (.h5)' --img_size 'input image size'
                            --number_of_classes 'number of classes in the dataset' --num_epochs number of epochs for training
                            --batch_size 'batch size for training' --freeze_feature_extractor 'default is false'
                            --train_from_scratch  'default is false' --extra_dense_layer 'default is false'
```
#### Evaluation classifier results
```commandline
python3 evaluate_classifier.py -h  'to see help'

python3 evaluate_classifier.py  --model_path 'full path to trained keras model' --data_dir 'full path to test images, expects a folder with sub-folder for each class'
```




### To train CNN and AE images folder must have next structure
```bash
├── images
│   ├── class_1
│   │   ├── img_1.jpg
│   │   ├── img_2.jpg
│   │   ├── ...
│   │   └── img_n.jpg
│   ├── class_2
│   │   ├── img_1.jpg
│   │   ├── img_2.jpg
│   │   ├── ...
│   │   └── img_n.jpg
│   ...
│   └── class_n
│   │   ├── img_1.jpg
│   │   ├── img_2.jpg
│   │   ├── ...
│   │   └── img_n.jpg
└──
```