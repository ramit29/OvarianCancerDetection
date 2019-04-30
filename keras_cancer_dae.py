import numpy as np
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import (Input, Dense, Concatenate)
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split

from keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler
# input image dimensions
img_rows, img_cols = 28, 28
input_shape = (3922, )
#cancer = open("cluster_features/Normal_Early/Hierarchical_200_Normal_Early.csv","r")
cancer = open("early_normal.csv","r")
next(cancer)
cancer = cancer.readlines()
cancer_main=[]
for i in cancer:
    i=i.strip("\n").split(",")
    if i[1] == '1':
        print(i[1])
        i[1] = 1
    else:
        i[1] = 0
    cancer_main.append(i)
cancer_x=[]
for i in cancer_main:
    cancer_x.append(i[3:])
cancer_met=[]
for i in cancer_x:
    cancer_met.append([float(j) for j in i])
cancer_target=[]
for i in cancer_main:
    cancer_target.append(i[1])

scaling = MinMaxScaler(feature_range=(-1,1)).fit(cancer_met)
#X_train = scaling.transform(X_train)
#X_test = scaling.transform(X_test)
cancer_met = scaling.transform(cancer_met)

# the data, shuffled and split between train and test sets
x_train, x_test, y_train, y_test = train_test_split(cancer_met, cancer_target, test_size=0.2, random_state=0)
x_train = np.array(x_train)
x_test = np.array(x_test)



noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=np.shape(x_train))
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=np.shape(x_test))
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)



x_feat_train = np.concatenate((x_train, x_test), axis=0)
x_feat_train_noisy = np.concatenate((x_train_noisy, x_test_noisy), axis=0)

print(x_feat_train_noisy.shape[0], ' dae train samples')

def DEEP_DAE(features_shape, act='relu'):

    # Input
    x = Input(name='inputs', shape=features_shape, dtype='float32')
    o = x
    o = Dense(512, activation=act, name='dense1')(o)
    o = Dense(512, activation=act, name='dense2')(o)
    o = Dense(512, activation=act, name='dense3')(o)
    dec = Dense(3922, activation='sigmoid', name='dense_dec')(o)

# Print network summary
    Model(inputs=x, outputs=dec).summary()

    return Model(inputs=x, outputs=dec)



batch_size = 32
epochs = 40

autoenc = DEEP_DAE(input_shape)
autoenc.compile(optimizer='adadelta', loss='binary_crossentropy')

autoenc.fit(x_feat_train_noisy, x_feat_train, epochs=epochs,
        batch_size=batch_size, shuffle=True)



def FEATURES(model):
    input_ = model.get_layer('inputs').input
    feat1 = model.get_layer('dense1').output
    feat2 = model.get_layer('dense2').output
    feat3 = model.get_layer('dense3').output
    feat = Concatenate(name='concat')([feat1, feat2, feat3])
    model = Model(inputs=[input_],
                      outputs=[feat1])
    return model

_model = FEATURES(autoenc)
features_train = _model.predict(x_train)
features_test = _model.predict(x_test)
print(features_train.shape, ' train samples shape')
print(features_test.shape, ' train samples shape')



def DNN(features_shape, num_classes, act='relu'):

    # Input
    x = Input(name='inputs', shape=features_shape, dtype='float32')
    o = x

    # Encoder / Decoder
    o = Dense(512, activation=act, name='dense1')(o)
    o = Dense(512, activation=act, name='dense2')(o)
    o = Dense(512, activation=act, name='dense3')(o)
    y_pred = Dense(num_classes, activation='sigmoid', name='pred')(o)

    # Print network summary
    Model(inputs=x, outputs=y_pred).summary()

    return Model(inputs=x, outputs=y_pred)



input_shape2 = (features_train.shape[1], )
num_classes = 2

y_train_ohe = np_utils.to_categorical(y_train, num_classes)
y_test_ohe = np_utils.to_categorical(y_test, num_classes)



batch_size = 128
epochs = 20
model_fname = 'dnn'

callbacks = [ModelCheckpoint(monitor='val_acc', filepath=model_fname + '.hdf5',
                             save_best_only=True, save_weights_only=True,
                             mode='min')]

deep = DNN(input_shape2, num_classes)
deep.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['acc'])

history = deep.fit(features_train, y_train_ohe, epochs=epochs,
                   batch_size=batch_size, shuffle=True,
                   validation_data=(features_test, y_test_ohe),
                   callbacks=callbacks)
