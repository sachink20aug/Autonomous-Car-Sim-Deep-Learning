import csv
import cv2
import os
import numpy as np
from keras import backend as K
from keras.layers import Cropping2D
import sklearn
import matplotlib.pyplot as plt
lines=[]
batch_size=8
with open('driving_log.csv') as csvfile:
	reader=csv.reader(csvfile)
	for line in reader:
		lines.append(line)
		

def generator(lines,batch_size=8):
    
    while 1:
        lines = sklearn.utils.shuffle(lines)
        for offset in range(0,len(lines),batch_size):
            batch_set=lines[offset:offset+batch_size]
            images=[]
            measurements=[]
            
            for line in batch_set:
                source_paths=[line[0],line[1],line[2]]
                filenames=[source_paths[0].split('/')[-1],source_paths[1].split('/')[-1],source_paths[2].split('/')[-1]]
                
                image_c=cv2.imread(filenames[0])
                image_c=cv2.cvtColor( image_c, cv2.COLOR_BGR2RGB);
                #resize(image_c, image_c, Size(64, 64,3));
                
                image_l=cv2.imread(filenames[1])
                image_l=cv2.cvtColor( image_l, cv2.COLOR_BGR2RGB);
                #resize(image_l, image_l, Size(64, 64,3));
                
                image_r=cv2.imread(filenames[2])
                image_r=cv2.cvtColor(image_r, cv2.COLOR_BGR2RGB);
                #resize(image_r, image_r, Size(64, 64,3));
                
                images.extend((image_c,image_l,image_r))
                images.extend((cv2.flip(image_c,1),cv2.flip(image_l,1),cv2.flip(image_r,1)))
                
                measurement=float(line[3])
                measurement_r=measurement-0.2
                measurement_l=measurement+0.2
                measurements.extend((measurement,measurement_l,measurement_r))
                measurements.extend((measurement*-1.0,measurement_l*-1.0,measurement_r*-1.0))
               
        
            X_train=np.array(images)
            y_train=np.array(measurements)
            
            yield sklearn.utils.shuffle(X_train, y_train)
        

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

train_generator = generator(train_samples, batch_size)
validation_generator = generator(validation_samples, batch_size)

               
from keras.models import Sequential
from keras.layers import Dense,Flatten,Lambda,Convolution2D
from keras.layers.pooling import MaxPooling2D

model=Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Convolution2D(24,5,5,activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid'))
model.add(Convolution2D(36,5,5,activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid'))
model.add(Convolution2D(48,5,5,activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid'))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse',optimizer='adam')
#model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=2)
history_object=model.fit_generator(train_generator, samples_per_epoch= len(train_samples)*6, validation_data=validation_generator,nb_val_samples=len(validation_samples)*6, nb_epoch=5)

print(history_object.history.keys())
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
model.save('model.h5')
K.clear_session()


	
