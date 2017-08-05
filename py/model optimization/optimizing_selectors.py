# -*- coding: utf-8 -*-
import numpy as np

from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import SGD, RMSprop, Adam


from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional, randint

def single_piece_selector_data(bs, ms, piece):
    pieces = ['Pawn', 'Rook', 'Knight', 'Bishop', 'Queen', 'King']
    move_selector = ms[:, 1]
    piece_bs = bs[move_selector == piece]
    piece_ms = ms[move_selector == piece, 4:6]
    y = np.apply_along_axis(func1d=convert_coord, axis=1, arr=piece_ms)
    print("The {} Move Selector data set contains {} boards".format(pieces[int(piece)], piece_ms.shape[0]))
    return piece_bs.astype('int'), y.reshape(y.shape[0], 64)

def ms_train_test_data():
    def load_data():
        print('Loading data...')
        loaded = np.load('../data/A00-139_first_10000.npz')
        boards = loaded['arr_0']
        targets = loaded['arr_1']
        print('Data loaded.  Splitting data...')
        b_train, b_test, t_train, t_test = train_test_split(boards, targets, test_size=0.25)
        print('Returning output...')
        return b_train, b_test, t_train, t_test

    def convert_colors(b, t):
        def b_to_w(boards):
            return np.fliplr(boards)*-1
        
        move_made_by = t[:, 0]
        bbs = b[(move_made_by == "b")]
        bms = t[(move_made_by == 'b')]
        wbs = b[(move_made_by == "w")]
        wms = t[(move_made_by == 'w')]
        bbs_t = np.array([b_to_w(b) for b in bbs])
        bs = np.concatenate([bbs_t, wbs], axis=0)
        ms = np.concatenate([bms, wms], axis=0)
        return bs, ms

    def move_selector_data(bs, ms): 
        def convert_coord(a):
            board = np.zeros((8, 8))
            board[int(a[0]),int(a[1])] = 1
            return board

        y = np.apply_along_axis(func1d=convert_coord, axis=1, arr=ms[:, 2:4])
        print("The Move Selector data set contains {} boards".format(y.shape[0]))
        return bs.astype('int'), y.reshape(y.shape[0], 64)

    b_train, b_test, t_train, t_test = load_data()
    X_train, y_train = convert_colors(b_train, t_train)
    ms_X_tr, ms_y_tr = move_selector_data(X_train, y_train)
    X_test, y_test = convert_colors(b_test, t_test)
    ms_X_test, ms_y_test = move_selector_data(X_test, y_test)
    return ms_X_tr, ms_X_test, ms_y_tr, ms_y_test

def move_selector_model(X_train, X_test, y_train, y_test):
    BOARD_CHANNELS = 6
    BOARD_ROWS = 8
    BOARD_COLS = 8    
    NB_CLASSES = 64
    NB_EPOCH = 10
    BATCH_SIZE = 128

    inputs = Input(shape=(BOARD_CHANNELS, BOARD_COLS, BOARD_ROWS))
    
    x = Conv2D(32, (3, 3), padding='same', activation="relu")(inputs)
    x = Flatten()(x)
    
    x = Dense({{choice([32, 64])}}, activation="relu", name="Dense")(x)
    
    predictions = Dense(NB_CLASSES, activation="softmax", name="Output")(x)
    
    model = Model(inputs=[inputs], outputs=[predictions])
    
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics = ["accuracy"])
    
    model.fit(X_train, y_train,
              batch_size=BATCH_SIZE, epochs=NB_EPOCH,
              verbose=0,
              validation_data=(X_test, y_test))
    
    score, acc = model.evaluate(X_test, y_test, verbose=0)
    
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=move_selector_model,
                                          data=ms_train_test_data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())
    print(best_run)

#    X_train, X_test, y_train, y_test = ms_train_test_data()
#    fit_model = move_selector_model(X_train, X_test, y_train, y_test)
#    print(fit_model['loss'])