import csv;
import cv2;
import numpy as np;
import os;

lines=[]
with open('../Training/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile);
    for line in reader:
        lines.append(line);

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

import sklearn;
def generator(samples, batch_size=32):
    num_samples = len(samples);
    while 1:
        sklearn.utils.shuffle(samples);
        for offset in range(0, num_samples, int(batch_size/2)):
            #print("Offset: ",offset);
            batch_samples = samples[offset:offset+int(batch_size/2)];
            images=[]
            measurements=[]
            for line in batch_samples:
                source_path = line[0];
                filename = source_path.split('/')[-1]
                current_path = '../Training/IMG/'+filename;
                image = cv2.imread(current_path)
                images.append(image);
                measurement=float(line[3])
                measurements.append(measurement);
                # adding flipped image to image set
                flipped_image=np.fliplr(image);
                measurement_flipped = -measurement;
                images.append(flipped_image);
                measurements.append(measurement_flipped);

            X_train=np.array(images);
            y_train=np.array(measurements);
            #print("X_train size: ",len(X_train));
            yield sklearn.utils.shuffle(X_train, y_train);

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Model
from keras.models import Sequential;
from keras.layers import Flatten, Dense, Lambda, Cropping2D;
from keras.layers.convolutional import Convolution2D;
from keras.layers.pooling import MaxPooling2D;

model = Sequential();
model.add(Lambda(lambda x: x/255.0 - .5, input_shape=(160,320,3)));
model.add(Cropping2D(cropping=((70,25),(0,0))));
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"));
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"));
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"));
#model.add(MaxPooling2D());
model.add(Convolution2D(64,3,3,activation="relu"));
#model.add(MaxPooling2D());
model.add(Convolution2D(64,3,3,activation="relu"));
#model.add(MaxPooling2D());
#model.add(MaxPooling2D());
model.add(Flatten());
model.add(Dense(100));
model.add(Dense(50));
model.add(Dense(10));
model.add(Dense(1));

model.compile(loss='mse', optimizer='adam');
#model.fit(X_train, y_train, validation_split=.2, shuffle=True, nb_epoch=4);
import matplotlib.pyplot as plt;
history_object = model.fit_generator(train_generator, samples_per_epoch =2*len(train_samples),
        validation_data = validation_generator,nb_val_samples = 2*len(validation_samples), 
                nb_epoch=4, verbose=1);
model.save('model.h5');
### print the keys contained in the history object
print(history_object.history.keys())
### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

