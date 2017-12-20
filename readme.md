# Vehicle Detection -- UMICH ROB 599 Final Project

## Dependencies:

Windows
opencv
numpy
darknet

## Method:

Please refer to the PDF file for details.

## Procedure to reproduce our result:

* download the dataset from [ROB599 Dataset] (http://umich.edu/~fcav/rob599_dataset_deploy.zip) and extract to /deploy
* run main.py to preprocess data
* run train/obj/pprocess.py to generate list for validation set
* download [pretrained weight file] (http://pjreddie.com/media/files/darknet19_448.conv.23) to /train
* run train/train.bat to start training. This may take 5~8 hours
* follow instruction from [darknet repo] (https://github.com/AlexeyAB/darknet) to decide which weight to use
* run train/obj_test.bat to generate raw results. You may need to modify the path to the weight file.
* run train/translate_yolo_finetune.py to generate final .csv file.

## Preview of the prediction of our network:

![prediction] (predictions.jpg)

## Our result:

On the [kaggle competetion] of the University of Michigan ROB 599 final project, we ranked 6 in the final.
