from keras.models import Sequential
from keras.layers import Lambda
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.callbacks import ModelCheckpoint, CSVLogger
from sklearn.model_selection import train_test_split
import csv
import config as cfg
from prepare_data import batch_data_gen
import math

def nvidia_nn_model():
    
    model = Sequential()
    xavier_init = "glorot_uniform"
    # standardize input
    model.add(Lambda(lambda xx: xx / 127.5 - 1., input_shape=(cfg.h, cfg.w, 3)))
    model.add(Conv2D(24, (5, 5), padding="valid", strides=(2, 2), kernel_initializer=xavier_init))
    model.add(Activation('elu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(36, (5, 5), padding="valid", strides=(2, 2), kernel_initializer=xavier_init))
    model.add(Activation('elu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(48, (5, 5), padding="valid", strides=(2, 2), kernel_initializer=xavier_init))
    model.add(Activation('elu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), padding="valid", kernel_initializer=xavier_init))
    model.add(Activation('elu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), padding="valid", kernel_initializer=xavier_init))
    model.add(Activation('elu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(100, kernel_initializer=xavier_init))
    model.add(Activation('elu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, kernel_initializer=xavier_init))
    model.add(Activation('elu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, kernel_initializer=xavier_init))
    model.add(Activation('elu'))
    model.add(Dense(1, kernel_initializer=xavier_init))
    model.summary()

    return model

if __name__ == '__main__':

    # split the data into training and validation sets
    with open('./data/driving_log.csv', 'r') as f:
        reader = csv.reader(f)
        driving_data = [row for row in reader][1:]

    train_data, val_data = train_test_split(driving_data, test_size=0.2, random_state=1)

    # get the model
    nvidia_nn = nvidia_nn_model()
    
    # save the architecture of a model
    with open('model.json', 'w') as f:
        f.write(nvidia_nn.to_json())
        
    # compile the model
    nvidia_nn.compile(optimizer='adam', loss='mse')

    # to save history
    history_log = CSVLogger(filename='logs/history.csv')

    # to save weights
    checkpoints = ModelCheckpoint('checkpoints/weights.{epoch:02d}-{val_loss:.5f}.hdf5')

    # fit model
    nvidia_nn.fit_generator(generator=batch_data_gen(train_data, data_dir='/opt/carnd_p3/data'),
                            steps_per_epoch=math.ceil(len(train_data)/cfg.batch_size),
                            epochs=50,
                            validation_data=batch_data_gen(val_data, data_dir='/opt/carnd_p3/data'),
                            validation_steps=math.ceil(len(val_data)/cfg.batch_size),
                            callbacks=[checkpoints, history_log])
    
    # save the architecture of the model, weights, the training configuration (loss, optimizer) etc.
    # nvidia_nn.save('model.h5') 
    # save the weights of a model
    # nvidia_nn.save_weights('model_weights.h5')
    