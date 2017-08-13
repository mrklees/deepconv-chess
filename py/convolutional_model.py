import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# Building the Neural Network
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam

'''
Functions for Loading Data
'''
# () => ndarray, ndarray, ndarray, ndarray
# Loads data from a numpy npz file. Then splits the data set into training
# and test sets.
def load_data():
    print('Loading data...')
    loaded = np.load('data/A00-139_first_10000.npz')
    boards = loaded['arr_0']
    targets = loaded['arr_1']
    print('Data loaded.  Splitting data...')
    b_train, b_test, t_train, t_test = train_test_split(boards, targets, test_size=0.25)
    print('Returning output...')
    return b_train, b_test, t_train, t_test

# Performs the transformation on input board to reverse white and black
# position.  np.fliplr is a reflection along the 45 degree 
def b_to_w(boards):
    return np.fliplr(boards)*-1

# Makes corresponding changes to the target data for changing black
# and white position. 
def convert_colors(b, t):
    move_made_by = t[:, 0]
    bbs = b[(move_made_by == "b")]
    bms = t[(move_made_by == 'b')]
    wbs = b[(move_made_by == "w")]
    wms = t[(move_made_by == 'w')]
    bbs_t = np.array([b_to_w(b) for b in bbs])
    bs = np.concatenate([bbs_t, wbs], axis=0)
    ms = np.concatenate([bms, wms], axis=0)
    return bs, ms

# Converts coordinates to a board state
def convert_coord(a):
    #print(a)
    board = np.zeros((8, 8))
    board[int(a[0]),int(a[1])] = 1
    return board

# Extracts the data needed for the move selector model
def move_selector_data(bs, ms):
    # piece selector data consists of all available board positions. The predictor is the index of the piece
    # that moved (0 through 5).  
    y = np.apply_along_axis(func1d=convert_coord, axis=1, arr=ms[:, 2:4])
    #print(y.shape)
    print("The Move Selector data set contains {} boards".format(y.shape[0]))
    return bs.astype('int'), y.reshape(y.shape[0], 64)

# Extracts the data needed for a particular piece's selector. 
def single_piece_selector_data(bs, ms, piece):
    pieces = ['Pawn', 'Rook', 'Knight', 'Bishop', 'Queen', 'King']
    move_selector = ms[:, 1]
    piece_bs = bs[move_selector == piece]
    piece_ms = ms[move_selector == piece, 4:6]
    y = np.apply_along_axis(func1d=convert_coord, axis=1, arr=piece_ms)
    print("The {} Move Selector data set contains {} boards".format(pieces[int(piece)], piece_ms.shape[0]))
    return piece_bs.astype('int'), y.reshape(y.shape[0], 64)
