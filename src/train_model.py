from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import InputLayer, Dense, Dropout, LayerNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split


def main():
    root_dir = str(Path(os.path.dirname(os.path.realpath(__file__))) / '..')
    pdata = np.loadtxt(root_dir + '/out/pdata.csv', delimiter=',')
    print('pdata shape : ', pdata.shape)
    plabel = np.ones(pdata.shape[0])
    print('plabel shape : ', plabel.shape)
    ndata = np.loadtxt(root_dir + '/out/ndata.csv', delimiter=',')
    print('ndata shape : ', ndata.shape)
    nlabel = np.zeros(ndata.shape[0])
    print('nlabel shape : ', nlabel.shape)

    X = np.append(pdata, ndata, axis=0)
    y = np.append(plabel, nlabel, axis=0)
    print('X shape : ', X.shape, 'y shpae : ', y.shape)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(InputLayer(input_shape=(30,)))
    model.add(LayerNormalization(axis=1))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid', name='Output'))

    model.summary()

    model.compile(optimizer=Adam(0.0002,), loss='binary_crossentropy', metrics='accuracy')

    checkpoint = ModelCheckpoint(filepath=root_dir + '/out/model/checkpoint.h5', verbose=1, save_best_only=True)
    earlyStopping = EarlyStopping(patience=10, restore_best_weights=True)

    history = model.fit(x_train, y_train, batch_size=128, epochs=500, validation_data=(x_val, y_val),
                        callbacks=[checkpoint, earlyStopping])

    hist_df = pd.DataFrame(history.history)
    hist_csv_file = root_dir + '/out/model/learning_curve.csv'
    with open(hist_csv_file, mode='w') as file:
        hist_df.to_csv(file)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(['train', 'validation'])

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend(['train', 'validation'])

    plt.show()

    print(model.evaluate(x_test, y_test))
    model.save(root_dir + '/out/model/dump_classifier')

if __name__ == '__main__':
    main()
