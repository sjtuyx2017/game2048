import numpy as np
import keras
import h5py
from keras.models import Model
from keras.layers import Dense,Conv2D,Flatten,Input,concatenate,Activation,BatchNormalization,Dropout
import pickle
import time


BATCH_SIZE = 100
NUM_CLASSES = 4
NUM_EPOCHS = 20

OUT_SHAPE = (4,4)
CAND = 16
map_table = {2**i: i for i in range(1,CAND)}
map_table[0] = 0

def one_hot_encoding(arr):
    result=np.zeros(shape=OUT_SHAPE+(CAND,),dtype=bool)
    for r in range(OUT_SHAPE[0]):
        for c in range(OUT_SHAPE[1]):
            result[r,c,map_table[arr[r,c]]]=1
    return result


if __name__ == '__main__':
    X_train = []
    X_test = []
    f1=open('./training_data/train_data.pkl','rb')
    f2=open('./training_data/train_label.pkl','rb')
    f3=open('./training_data/test_data.pkl','rb')
    f4=open('./training_data/test_label.pkl','rb')
    train_data = pickle.load(f1)
    train_label = pickle.load(f2)
    test_data = pickle.load(f3)
    test_label = pickle.load(f4)
    f1.close()
    f2.close()
    f3.close()
    f4.close()
    for item in train_data:
        X_train.append(one_hot_encoding(item))
    for item in test_data:
        X_test.append(one_hot_encoding(item))

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = keras.utils.to_categorical(train_label,num_classes = NUM_CLASSES)
    y_test = keras.utils.to_categorical(test_label,num_classes = NUM_CLASSES)


    inputs = Input((4,4,16))
    conv = inputs
    Filters = 128
    conv41 = Conv2D(filters=Filters, kernel_size=(4, 1), kernel_initializer='he_uniform')(conv)
    conv14 = Conv2D(filters=Filters, kernel_size=(1, 4), kernel_initializer='he_uniform')(conv)
    conv22 = Conv2D(filters=Filters, kernel_size=(2, 2), kernel_initializer='he_uniform')(conv)
    conv33 = Conv2D(filters=Filters, kernel_size=(3, 3), kernel_initializer='he_uniform')(conv)
    conv44 = Conv2D(filters=Filters, kernel_size=(4, 4), kernel_initializer='he_uniform')(conv)
    hidden = concatenate([Flatten()(conv41),Flatten()(conv14),Flatten()(conv22),Flatten()(conv33),Flatten()(conv44)])
    x = BatchNormalization()(hidden)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)

    for width in [512,128]:
        x = Dense(width,kernel_initializer='he_uniform')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)


    outputs = Dense(4,activation='softmax')(x)
    model = Model(inputs,outputs)

    model.summary()
    start = time.time()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
    model.save('model.h5')
    score_train = model.evaluate(X_train, y_train)
    print('Training loss: %.4f, Training accuracy: %.2f%%' % (score_train[0], score_train[1] * 100))
    score_test = model.evaluate(X_test, y_test)
    print('Testing loss: %.4f, Testing accuracy: %.2f%%' % (score_test[0], score_test[1] * 100))
    end = time.time()
    print("time:",end-start)
