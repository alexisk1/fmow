import os
import csv
import glob
import argparse
from sklearn.utils import shuffle
import lib.keras_resnet.resnet as resnet
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
import numpy as np
import json
from pprint import pprint
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.utils.np_utils import to_categorical

from keras.models import model_from_json

from pprint import pprint
parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_dir', type=str, default='',
    help='The directory where the FMoW input data is stored.')


FLAGS, unparsed = parser.parse_known_args()

train_samples = glob.glob(str(os.path.join(FLAGS.data_dir, 'train'))+"/*.jpg")

print(train_samples[0][0:-4] )
batch_sample=train_samples[0]
name = batch_sample
with open(batch_sample[0:-4]+'.json') as data_file:    
     data = json.load(data_file)
bounding_box = data ['bounding_boxes'][0]['box']
category = data ['bounding_boxes'][0]['category']
x1 = bounding_box[0]
y1 = bounding_box[1]
w = bounding_box[2]
h = bounding_box[3]
print(x1,y1,w ,h)
pprint(data)

from numpy import array
from numpy import argmax
from keras.utils import to_categorical
# define example
classes = ['background','airport', 'airport_hangar', 'airport_terminal', 'amusement_park', 'aquaculture', 'archaeological_site', 'barn', 'border_checkpoint', 'burial_site', 'car_dealership', 'construction_site', 'crop_field', 'dam', 'debris_or_rubble', 'educational_institution', 'electric_substation', 'factory_or_powerplant', 'fire_station', 'flooded_road', 'fountain', 'gas_station', 'golf_course', 'ground_transportation_station', 'helipad', 'hospital', 'impoverished_settlement', 'interchange', 'lake_or_pond', 'lighthouse', 'military_facility', 'multi-unit_residential', 'nuclear_powerplant', 'office_building', 'oil_or_gas_facility', 'park', 'parking_lot_or_garage', 'place_of_worship', 'police_station', 'port', 'prison', 'race_track', 'railway_bridge', 'recreational_facility', 'road_bridge', 'runway', 'shipyard', 'shopping_mall', 'single-unit_residential', 'smokestack', 'solar_farm', 'space_facility', 'stadium', 'storage_tank', 'surface_mine', 'swimming_pool', 'toll_booth', 'tower', 'tunnel_opening', 'waste_disposal', 'water_treatment_facility', 'wind_farm', 'zoo']

data = array(classes)
values = array(data)
print(values)
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)
# invert first example
print(label_encoder.transform(['zoo']))
#label_encoder.inverse_transform(inverted)
inverted = onehot_encoded[label_encoder.transform(['zoo'])]
print(inverted)

test_samples = glob.glob(str(os.path.join(FLAGS.data_dir, 'test'))+"/*.jpg")

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=10)
csv_logger = CSVLogger('resnet18_fmow.csv')


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np
import sklearn

ch, row, col = 3, 100, 100  # Trimmed image format

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=10)
csv_logger = CSVLogger('resnet18_fmow.csv')

batch_size = 32
nb_classes = 63
nb_epoch = 200
data_augmentation = True

# input image dimensions
img_rows, img_cols = 32, 32
# The CIFAR10 images are RGB.
img_channels = 3
nb_classes = 63


def res_generator(samples,model):
   num_samples = len(samples)
   pred=[]
   with open("test_results.txt",'w') as res_file: 
    for sample in samples:
        name = sample
        with open(sample[0:-4]+'.json') as data_file:    
              data = json.load(data_file)
        print(sample)
        for bound_box_info in data ['bounding_boxes']:
            bounding_box= bound_box_info['box']
            pprint(bound_box_info)
            x1 = bounding_box[0]
            y1 = bounding_box[1]
            w = bounding_box[2]
            h = bounding_box[3]
            if(w>0 and h>0):
               idx= bound_box_info['ID']
               image = cv2.imread(name)
               img  = image
               print("Before:", image.shape)
               image = image[y1:y1+h,x1:x1+w]
               print("after:", image.shape)
               if(1):
                  image = cv2.resize(image,(64, 64), interpolation=cv2.INTER_AREA)  
                  image =np.expand_dims(image, axis=0)
                  y_test = np.argmax(model.predict(image, batch_size=1, verbose=0))
                  print("PRED:", label_encoder.inverse_transform(y_test))
                  res_file.write(str(idx) +', '+ label_encoder.inverse_transform(y_test)+'\n')
               else:
                  print("PRED: FALSE")
                  res_file.write(idx,', FALSE\n')
   res_file.close()

with open("model.json") as json_file:    
     model = model_from_json(json_file.read())
json_file.close()
model.load_weights("./weights.best.hdf5", True)

res_generator(test_samples,model)

