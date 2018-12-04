
#-------------------------------------------------------------------------
# Note: please don't use any additional package except the following packages
import numpy as np
#-------------------------------------------------------------------------
'''
    Problem 1: TicTacToe and MiniMax 
    In this problem, you will implement a version of the TicTacToe game and a minimax player.
    You could test the correctness of your code by typing `nosetests -v test1.py` in the terminal.
'''

#-------------------------------------------------------
class PlayerRandom:
    '''a random player, which chooses valid moves randomly. '''
    # ----------------------------------------------
    def play(self,s,x=1):
        '''
           The policy function, which chooses one move in the game.  
           Here we choose a random valid move.
           Input:
                s: the current state of the game, an integer matrix of shape 3 by 3. 
                    s[i,j] = 0 denotes that the i-th row and j-th column is empty
                    s[i,j] = 1 denotes that the i-th row and j-th column is taken by "X". 
                    s[i,j] = -1 denotes that the i-th row and j-th column is taken by the "O".
               x: the role of the player, 1 if you are the "X" player in the game
                    -1 if you are the "O" player in the game. 
           Outputs:
                r: the row number, an integer scalar with value 0, 1, or 2. 
                c: the column number, an integer scalar with value 0, 1, or 2. 
        '''
        empty = []
        for i in range(3):
            for j in range(3):
                if s[i][j] == 0:
                    empty.append((i, j))

        move = np.random.randint(len(empty))
        r, c = empty[move]

        return r, c


#-------------------------------------------------------
class TicTacToe:
    '''TicTacToe is a game engine. '''
    # ----------------------------------------------
    def __init__(self):
        ''' Initialize the game. 
            Input:
                self.s: the current state of the game, a numpy integer matrix of shape 3 by 3. 
                        self.s[i,j] = 0 denotes that the i-th row and j-th column is empty
                        self.s[i,j] = 1 denotes that the i-th row and j-th column is "X"
                        self.s[i,j] = -1 denotes that the i-th row and j-th column is "O"
        '''
        self.s = np.zeros((3,3))


    # ----------------------------------------------
    def play_x(self, r, c):
        '''
           X player takes one step with the location (row and column number)
            Input:
                r: the row number, an integer scalar with value 0, 1, or 2. 
                c: the column number, an integer scalar with value 0, 1, or 2. 
        '''
        assert  self.s[r,c]==0
        self.s[r, c] = 1

    # ----------------------------------------------
    def play_o(self, r, c):
        '''
           O player take one step with the location (row and column number)
            Input:
                r: the row number, an integer scalar with value 0, 1, or 2. 
                c: the column number, an integer scalar with value 0, 1, or 2. 
        '''
        assert  self.s[r,c]==0
        self.s[r, c] = -1


    # ----------------------------------------------
    @staticmethod
    def avail_moves(s):
        '''
           Get a list of avaiable (valid) next moves from the given state s of the game 
            Input:
                s: the current state of the game, an integer matrix of shape 3 by 3. 
                    s[i,j] = 0 denotes that the i-th row and j-th column is empty
                    s[i,j] = 1 denotes that the i-th row and j-th column is taken by "X" player. 
                    s[i,j] = -1 denotes that the i-th row and j-th column is taken by "O" player.
            Outputs:
                m: a list of possible next moves, where each next move is a (r,c) tuple, 
                   r denotes the row number, c denotes the column number. 
        '''
        m = []
        for i in range(3):
            for j in range(3):
                if s[i][j] == 0:
                    m.append((i, j))
        return m

    # ----------------------------------------------
    @staticmethod
    def check(s):
        '''
            check if the game has ended.  
            Input:
                s: the current state of the game, an integer matrix of shape 3 by 3. 
                    s[i,j] = 0 denotes that the i-th row and j-th column is empty
                    s[i,j] = 1 denotes that the i-th row and j-th column is taken by "X" player. 
                    s[i,j] = -1 denotes that the i-th row and j-th column is taken by "O" player.
            Outputs:
                e: the result, an integer scalar with value 0, 1 or -1.
                    if e = None, the game doesn't end yet.
                    if e = 0, the game is a draw.
                    if e = 1, X player won the game.
                    if e = -1, O player won the game.
        '''
        all_rows = []
        array = np.asarray(s)
        for i in range(3):
            row = array[i, :]
            column = array[:, i]
            all_rows.append(row.tolist())
            all_rows.append(column.tolist())

        diagonal1 = [array[i, i] for i in range(3)]
        all_rows.append(diagonal1)

        diagonal2 = [array[2, 0], array[1, 1], array[0, 2]]
        all_rows.append(diagonal2)

        if [1, 1, 1] in all_rows:
            return 1
        elif [-1, -1, -1] in all_rows:
            return -1

        # check if the game ends
        num_of_empty = np.count_nonzero(array == 0)
        return 0 if num_of_empty == 0 else None

    # ----------------------------------------------
    def game(self,x,o):
        '''
            run a tie-tac-toe game starting from the current state of the game, letting X and O players to play in
            turns.
            Here we assumes X player moves first in a game, then O player moves.
            let x player and o player take turns to play and after each move check if the game ends 
            Input:
                x: the "X" player (the first mover), such as PlayerRandom, you could call x.play() to let this player to
                choose ome move.
                o: the "O" player (the second mover)
            Outputs:
                e: the result of the game, an integer scalar with value 0, 1 or -1.
                    if e = 0, the game ends with a draw/tie.
                    if e = 1, X player won the game.
                    if e = -1, O player won the game.
        '''
        if self.check(self.s) is not None:
            return self.check(self.s)

        while(True):
            r, c = x.play(self.s, 1)
            self.play_x(r, c)
            if TicTacToe.check(self.s) is not None:
                return TicTacToe.check(self.s)

            r, c = o.play(self.s, -1)
            self.play_o(r, c)
            if TicTacToe.check(self.s) is not None:
                return TicTacToe.check(self.s)

            
#-----------------------------------------------
#   MiniMax Player
#-----------------------------------------------
class Node:
    '''
        Search Tree Node
        Inputs: 
            s: the current state of the game, an integer matrix of shape 3 by 3. 
                s[i,j] = 0 denotes that the i-th row and j-th column is empty
                s[i,j] = 1 denotes that the i-th row and j-th column is taken by "X". 
                s[i,j] = -1 denotes that the i-th row and j-th column is taken by the "O".
            x: the role of the player, 1 if you are the "X" player in the game
                    -1 if you are the "O" player in the game. 
    '''
    def __init__(self, s, x=1, c=None, m=None, v=None):
        self.s = s
        self.c = []     # a list of children nodes
        self.x = x      # the role of the player in the game (X player:1, or O player:-1)
        self.m = m      # the move that it takes from the parent node to reach this node.
                        # m is a tuple (r,c), r:row of move, c:column of th move
        self.v = v      # the value of the node (X player will win:1, tie: 0, lose: -1)

    # ----------------------------------------------
    def expand(self):
        '''
         Expand the current tree node for one-level.
         Add one child node for each possible next move in the game.
        '''
        # if the game in the current state has already ended,  return/exit
        if TicTacToe.check(self.s) is not None:
            return

        # if the game has not ended yet, expand the current node with one child node for each valid move 
        all_moves = TicTacToe.avail_moves(self.s)
        for r, c in all_moves:
            # update s
            s = np.copy(self.s)
            s[r, c] = self.x
            # v = TicTacToe.check(s)
            node = Node(s, x=-self.x, m=(r, c))
            self.c.append(node)


    # ----------------------------------------------
    def build_tree(self):
        '''
        Given a node of the current state of the game, build a fully-grown search tree without computing the score of
        each node.
        Hint: you could use recursion to build the search tree        
        '''
        # expand the current node
        self.expand()

        # recursively build a subtree from each child node
        for node in self.c:
            node.build_tree()


    # ----------------------------------------------
    def compute_v(self):
        '''
           compute score of the current node of a search tree using minimax algorithm
           Here we assume that the whole search-tree is fully grown, but no score on any node has been computed yet
           before calling this function.
           After computing value of the current node, assign the value to n.v and assign the optimal next move to node.p
        '''
        # if the game has already ended, the score of the node is the result of the game (X won: 1, tie: 0, or O won:-1)
        result = TicTacToe.check(self.s)
        if result is not None:
            self.v = result
            return

        # otherwise: compute scores/values of all children nodes - run the function for all its children
        # Hint: you could use recursion to solve this problem.
        for node in self.c:
            node.compute_v()

        # set the value of the current node with the value of the best move
        # Hint: depending on whether the current node is "X" or "O" player, you need to compute either max (if X player)
        # or min (O player) of the values among the children nodes
        values = np.array([node.v for node in self.c])
        print(values)
        if self.x == 1:
            self.v = np.max(values)
            self.p = self.c[np.argmax(values)].m
        if self.x == -1:
            self.v = np.min(values)
            self.p = self.c[np.argmin(values)].m



#-------------------------------------------------------
class PlayerMiniMax:
    '''
        Minimax player, who choose optimal moves by searching the subtree with min-max.  
    '''

    # ----------------------------------------------
    def play(self,s,x=1):
        '''
          The policy function of the minimax player, which chooses one move in the game.  
          We need to first build a tree rooted with the current state, and compute the values of all the nodes in the
          tree.
           Inputs:
                s: the current state of the game, an integer matrix of shape 3 by 3. 
                    s[i,j] = 0 denotes that the i-th row and j-th column is empty
                    s[i,j] = 1 denotes that the i-th row and j-th column is taken by "X". 
                    s[i,j] = -1 denotes that the i-th row and j-th column is taken by the "O".
                x: the role of the player, 1 if you are the "X" player in the game
                    -1 if you are the "O" player in the game. 
           Outputs:
                r: the row number, an integer scalar with value 0, 1, or 2. 
                c: the column number, an integer scalar with value 0, 1, or 2. 
        '''
        # build a search tree with the current state as the root node
        node = Node(s, x)
        node.build_tree()

        # compute value
        node.compute_v()

        # find the best next move
        r, c = node.p

        return r, c
