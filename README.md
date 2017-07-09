## Chess Data for Deep Learning

Welcome!  This project has two primary goals.  

1.  To teach an agent chess using as little hand-coded knowledge as possible using Keras and TensorFlow.  
2.  To create a community resource of preprocessed chess game data.

As I was researching and preparing to work on the first problem, I noticed that prepared chess game data for machine learning was largely unavailable in large quantities.  It is my hope that this project can serve as a resource for the community of chess engine developers, and make it easier for folks to try novel approaches using prepared data.  The chess engine itself is also open source, and the most up to date version will be [hosted on github](https://github.com/mrklees/deepconv-chess). 

In the project files you will find compressed numpy save files (.npz) including thousands of games which have been processed into numpy arrays. The board states are composed of 6 x 8 x 8 numpy arrays with 6 channels corresponding to the 6 different types of pieces. Each board also corresponds to a numpy array containing the player who is moving and what the next move was including an index corresponding the piece, and the row and column coordinate values. 

In the course of doing research for this project, I made use of several resources and papers which I list below. They reflect what I think are some of the most exciting approaches in the application of deep learning to chess, and I have learned a ton from them. Giraffe, for example, applies reinforcement learning and deep learning to the problem of chess, and currently ranks 186th on the CCRL chess engine ranking for all engines. If you have ideas, I would love to hear from you in the discussion board.   Thanks!

#### Built on the Shoulder of Giants
In working on this project I spent a considerable amount of time reading the magnificent content on [the chess programming wiki](https://chessprogramming.wikispaces.com/)

There are also several papers which I leaned on considerably in learning how to create this engine.

*  Oshri, Barak, and Nishith Khandwala. ["Predicting moves in chess using convolutional neural networks."](http://cs231n.stanford.edu/reports/2015/pdfs/ConvChess.pdf) (2016). 
*  Lai, Matthew. ["Giraffe: Using deep reinforcement learning to play chess."](https://arxiv.org/abs/1509.01549) arXiv preprint arXiv:1509.01549 (2015).

Finally, much gratitude to the folks at [KingBase](kingbase-chess.net). KingBase provides 1.9 million games of players > 2000 ELO in multiple formats including pgn.