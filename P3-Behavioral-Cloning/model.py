import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Dropout, Flatten, Activation, Cropping2D
from keras.layers.convolutional import Convolution2D
from utils import INPUT_SHAPE, batch_generator
import argparse
import os

np.random.seed(0)

def load_data(args):
    """
    Load training data and split it into training and validation set
    """
    data_df = pd.read_csv(os.path.join(args.data_dir, 'driving_log.csv'))

    X = data_df[['center', 'left', 'right']].values
    y = data_df['steering'].values

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0)

    return X_train, X_valid, y_train, y_valid


def nvidia_model_1(args):
    """
    Return a modified NVIDIA model
    """
    print (INPUT_SHAPE)
    model = Sequential()
    model.add(Lambda(lambda x: x/255-0.5, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))  # subsample is stride
    model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Dropout(args.keep_prob))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()

    return model

def lenet_model(input_shape):
    """
    Return a Lenet model
    """
    model = Sequential()

    model.add(Cropping2D(cropping=((6, 25), (0, 0)), input_shape=INPUT_SHAPE, name="crop"))
    model.add(Lambda(lambda x: x/255-0.5, name="normalization"))
    model.add(Conv2D(32, 5, 5, activation='elu', name="conv_1"))
    model.add(Conv2D(64, 3, 3, activation='elu', name="conv_2"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="maxpool_1"))
    model.add(Dropout(0.5, name="dropout_1"))
    model.add(Flatten(name="flatten_1"))
    model.add(Dense(128, activation='elu', name="fc_1"))
    model.add(Dropout(0.5, name="dropout_2"))
    model.add(Dense(1, name="output"))
    return model

def nvidia_model_2(input_shape):
    """
    Return a NVIDIA model
    """
    model = Sequential()
    model.add(Lambda(lambda x: (x / 127.5 - 1.00), input_shape=INPUT_SHAPE))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), init='he_normal', activation='elu'))
    # model.add(Dropout(0.1))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), init='he_normal', activation='elu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), init='he_normal', activation='elu'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), init='he_normal', activation='elu'))
    # model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='elu', init='he_normal'))
    # model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='elu', init='he_normal'))
    model.add(Dropout(0.5))
    # model.add(Dense(50, activation='elu', init='he_normal'))
    model.add(Dense(10, activation='elu', init='he_normal'))
    # model.add(Dropout(0.5))
    model.add(Dense(1, init='he_normal'))
    return model


def train_model(model, args, X_train, X_valid, y_train, y_valid):
    """
    Train the model
    """
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=args.save_best_only,
                                 mode='auto')

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate))

    model.fit_generator(batch_generator(args.data_dir, X_train, y_train, args.batch_size, True),
                        args.samples_per_epoch,
                        args.nb_epoch,
                        max_q_size=1,
                        validation_data=batch_generator(args.data_dir, X_valid, y_valid, args.batch_size, False),
                        nb_val_samples=len(X_valid),
                        callbacks=[checkpoint],
                        verbose=1)


def s2b(s):
    """
    Converts a string to boolean value
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'


def main():
    """
    Load train/validation data set and train the model
    """
    parser = argparse.ArgumentParser(description='Behavioral Cloning Project')
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='..\\..\\datasets_collection\\behavior_cloning')
    parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',          type=int,   default=10)
    parser.add_argument('-s', help='samples per epoch',     dest='samples_per_epoch', type=int,   default=20000)
    parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=1)
    parser.add_argument('-o', help='save best models only', dest='save_best_only',    type=s2b,   default='true')
    parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=1.0e-4)
    args = parser.parse_args()
    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} -> {}'.format(key, value))
    print('-' * 30)

    data = load_data(args)
    model = lenet_model(args) 
    # train_model(model, args, *data)


if __name__ == '__main__':
    main()