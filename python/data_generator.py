import os
import re
from collections import defaultdict
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

import chess
from chess.pgn import read_game

import numpy as np

def extract_moves(game):
    # Takes a game from the pgn and creates list of the board state and the next
    # move that was made from that position.  The next move will be our 
    # prediction target when we turn this data over to the ConvNN.
    positions = list()
    board = chess.Board()
    moves = list(game.main_line())
    for move in moves:
        position, move_code = board.fen(), move.uci()
        positions.append([position, move_code])
        board.push(move)     
    return positions

def replace_nums(line):
    # This function cycles through a string which represents one line on the
    # chess board from the FEN notation.  It will then swap out the numbers
    # for an equivalent number of spaces.
    return ''.join([' '*8 if h=='8' else ' '*int(h) if h.isdigit() else'\n'if h=='/'else ''+h for h in line])
    
def split_fen(fen):
    # Takes the fen string and splits it into its component lines corresponding
    # to lines on the chess board and the game status. 
    fen_comps = fen.split(' ', maxsplit = 1)
    board = fen_comps[0].split('/')
    status = fen_comps[1]
    board = [replace_nums(line) for line in board]
    return board, status

def list_to_matrix(board_list):
    # Converts a list of strings into a numpy array by first 
    # converting each string into a list of its characters. 
    pos_list = [list(line) for line in board_list]
    return np.array(pos_list)

def channelize(mat):
    # processes a board into a 8 x 8 x 6 matrix where there is a 
    # channel for each type of piece.  1's correspond to white, and 
    # -1's correpond to black.
    output = np.empty([8, 8, 6])
    wpcs = ['P', 'R', 'N', 'B', 'Q', 'K']
    bpcs = ['p', 'r', 'n', 'b', 'q', 'k']
    positions = [np.isin(mat, pc).astype('int') - np.isin(mat, bpcs[i]).astype('int') for i, pc in enumerate(wpcs)]
    return np.stack(positions)

def uci_to_coords(uci):
    def conv_alpha_num(alpha):
        num = ord(alpha) - 97
        return num
    
    # Every UCI is a 4 character code indicated the from and to squares
    fc, fr = uci[0:2]
    tc, tr = uci[2:4]
    
    return [8-int(fr), conv_alpha_num(fc)], [8-int(tr), conv_alpha_num(tc)]

def process_status(status):
    # The last combination of characters in the FEN notation convey some different pieces of information
    # like the player who is to move next, and who can still castle. 
    # I have written the code to extract all of the different pieces, but the Agent will only need to know next_to_move. 
    splt = status.split(" ")
    next_to_move = splt[0]
    castling = splt[1]
    en_passant = splt[2]
    half_clock = splt[3]
    full_clock = splt[4]
    return next_to_move

def process_game(positions):
    # Takes a single game from a pgn and produces a dict of dicts which contains 
    # the board state, the next player to move, and the what the next move was (the prediction task).
    boards = []
    next_to_move = []
    for position in positions:
        board, status = split_fen(position[0])
        orig, dest = uci_to_coords(position[1])
        arrays = channelize(list_to_matrix(board))
        boards.append(arrays)
        piece_moved = [i for (i, mat) in enumerate(arrays) if (mat[int(orig[0]), int(orig[1])] == 1) | (mat[int(orig[0]), int(orig[1])] == -1)]
        if piece_moved == []:
            piece_moved = -1
        else:
            piece_moved = piece_moved[0]
        next_to_move.append([process_status(status), piece_moved, orig[0], orig[1], dest[0], dest[1]])
    try:
        boards, ntm = np.stack(boards), np.stack(next_to_move)
    except:
        return [], []
    return boards, ntm

def read_and_process(iteration):
    gm = read_game(pgn)
    positions = extract_moves(gm)
    boards, next_to_move = process_game(positions)
    #print("".join(["Completed: ", str(iteration),]))
    return boards, next_to_move

def wrangle_data_ip(num_games=10000, save_file=False):
    pool = ThreadPool(12) # Its even shorter than the single threaded version! Well... minus the other function I had to write...
    results = pool.map(read_and_process, range(num_games)) #Runs into a problem which will kill a small percentage of your games.
    pool.close() # But its totally worth it
    pool.join() # lol (I'll figure it out eventually...)
    return results

def wrangle_data(num_games=10000, save_file=False):
    # Meta process for data extraction in serial.. See above for parallelized version!
    boards, next_to_move = read_and_process(0)
    for i in range(1, num_games):
        new_boards, new_next_to_move = read_and_process(i)
        boards, next_to_move = np.concatenate((boards, new_boards), axis=0), np.concatenate((next_to_move, new_next_to_move), axis=0)
    if save_file:
        np.savez_compressed('first_{}_games'.format(num_games), results)
    return boards, next_to_move

def ip_results_to_np(results):
    # Splits a list of tuples into two lists.  Also filters out any errors which wrote as []'s. 
    boards = [result[0] for result in results if isinstance(result[0], np.ndarray)]
    targets = [result[1] for result in results if isinstance(result[1], np.ndarray)]
    # Then returns the full lists concatenated together
    return np.concatenate(boards, axis=0), np.concatenate(targets, axis=0)

if __name__ == "__main__":
    with open('../data/KingBase2017-A00-A39.pgn', encoding='latin1') as pgn:
        num_games=50000
        print("Recording the first {} games as matrices...".format(num_games))
        results = wrangle_data_ip(num_games=num_games, save_file=True)
        boards, targets = ip_results_to_np(results)
        print("Writing {} positions to file".format(boards.shape[0]))
        np.savez_compressed('../data/A00-139_first_{}'.format(num_games), boards, targets)