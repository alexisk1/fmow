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
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
import math 
parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_dir', type=str, default='',
    help='The directory where the FMoW input data is stored.')

parser.add_argument(
    '--data_aug', type=bool, default=False,
    help='Enable data augmentantion.')

FLAGS, unparsed = parser.parse_known_args()

train_samples = glob.glob(str(os.path.join(FLAGS.data_dir, 'train'))+"/*.jpg")

from numpy import array
from numpy import argmax
from keras.utils import to_categorical
# define example
classes = ['background','airport', 'airport_hangar', 'airport_terminal', 'amusement_park', 'aquaculture', 'archaeological_site', 'barn', 'border_checkpoint', 'burial_site', 'car_dealership', 'construction_site', 'crop_field', 'dam', 'debris_or_rubble', 'educational_institution', 'electric_substation', 'factory_or_powerplant', 'fire_station', 'flooded_road', 'fountain', 'gas_station', 'golf_course', 'ground_transportation_station', 'helipad', 'hospital', 'impoverished_settlement', 'interchange', 'lake_or_pond', 'lighthouse', 'military_facility', 'multi-unit_residential', 'nuclear_powerplant', 'office_building', 'oil_or_gas_facility', 'park', 'parking_lot_or_garage', 'place_of_worship', 'police_station', 'port', 'prison', 'race_track', 'railway_bridge', 'recreational_facility', 'road_bridge', 'runway', 'shipyard', 'shopping_mall', 'single-unit_residential', 'smokestack', 'solar_farm', 'space_facility', 'stadium', 'storage_tank', 'surface_mine', 'swimming_pool', 'toll_booth', 'tower', 'tunnel_opening', 'waste_disposal', 'water_treatment_facility', 'wind_farm', 'zoo']

data = array(classes)
values = array(data)

# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)

# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

# invert first example

#label_encoder.inverse_transform(inverted)



validation_samples = glob.glob(str(os.path.join(FLAGS.data_dir, 'val'))+"/*.jpg")

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=10)
csv_logger = CSVLogger('resnet18_fmow.csv')



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

import matplotlib.pyplot as plt
def generator(samples,image_datagen, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, int(batch_size)):
            batch_samples = samples[offset:offset+int(batch_size)]
            images = []
            pred = []
            for batch_sample in batch_samples:
                name = batch_sample
                with open(batch_sample[0:-4]+'.json') as data_file:    
                     data = json.load(data_file)
                bounding_box = data ['bounding_boxes'][0]['box']
                x1 = bounding_box[0]
                y1 = bounding_box[1]
                w = bounding_box[2]
                h = bounding_box[3]
                case = np.random.randint(0,8)
                if(case<3):
                  center_image = cv2.imread(name)
                  center_image = center_image[y1:y1+h,x1:x1+w]
                  center_image = cv2.resize(center_image,(64, 64), interpolation=cv2.INTER_AREA)

                  images.append(center_image)
                elif(case!=7):
                  center_image =cv2.imread(name)
                  center_image = center_image[y1:y1+h,x1:x1+w]
                  center_image = cv2.resize(center_image,(64, 64), interpolation=cv2.INTER_AREA)  

                  images.append(center_image)
                else:
                  center_image =cv2.imread(name)
                  center_image = center_image[y1+h:center_image.shape[0],x1+w:center_image.shape[1]]
                  center_image = cv2.resize(center_image,(64, 64), interpolation=cv2.INTER_AREA)  

                  images.append(center_image)
                category = label_encoder.transform([data ['bounding_boxes'][0]['category']])    
                pred.append(onehot_encoded[category])
            X_train = np.array(images)
            y_train = np.array(pred )
            y_train = np.reshape(y_train,(len(batch_samples), nb_classes))
            if( FLAGS.data_aug):
               #image_datagen.fit(X_train)
               X_train, y_train = next( image_datagen.flow(X_train, y_train, batch_size=32,save_to_dir='./aug/'))
            yield sklearn.utils.shuffle(X_train, y_train)


data_gen_args = dict(featurewise_center=False,
                     featurewise_std_normalization=False,
                     rotation_range=90.,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2,
                     horizontal_flip=True,
                     vertical_flip=True)
image_datagen = ImageDataGenerator(**data_gen_args)


seed = 7

# learning rate schedule
def step_decay(epoch):
	initial_lrate = 0.001
	drop = 0.5
	epochs_drop = 10.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

# compile and train the model using the generator function
train_generator = generator(train_samples,image_datagen, batch_size=32)
validation_generator = generator(validation_samples, image_datagen, batch_size=32)

#train_generator= image_datagen.flow(generator(train_samples, batch_size=32), augment=True, seed=seed)
#validation_generator = image_datagen.flow(generator(validation_samples, batch_size=32), augment=True, seed=seed)

model = resnet.ResnetBuilder.build_resnet_18((3, 64, 64), nb_classes)

model.summary() 
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy','precision','recall'])

filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
lrate = LearningRateScheduler(step_decay)

callbacks_list = [checkpoint]
# Fit the model
model.fit_generator(train_generator, steps_per_epoch=int(len(train_samples)/32), validation_data=validation_generator,
                    validation_steps=int(len(validation_samples)/32), epochs=6, callbacks=callbacks_list)

model.save_weights("./model2.h5", True)
model_json = model.to_json()
json_file=open("model2.json", "w")
json_file.write(model_json)
