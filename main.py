from preprocessing_for_yolo_rep_channel import preprocessor
from preprocessing_for_yolo_rep_channel_test import preprocessor_test

p = preprocessor()
p.generateAllData()

pt = preprocessor_test()
pt.generateAllData()
