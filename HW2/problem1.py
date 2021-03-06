#-------------------------------------------------------------------------
'''
    Problem 1: Elo ranking algorithm 
    In this problem, you will implement a version of the Elo rating algorithm.
    You could test the correctness of your code by typing `nosetests -v test1.py` in the terminal.
    To check all the codes in your submission, please type `nosetests -v' in the terminal.
'''

#--------------------------
def compute_EA(RA, RB):
    '''
        compute the expected probability of player A to win in a game with player B.
        Input:
            RA: the rating of player A, a float scalar value
            RB: the rating of player B, a float scalar value
        Output:
            EA: the expected probability of A wins, a float scalar value between 0 and 1.
    '''
    EA = 1 / (1 + 10 ** ((RB-RA)/400))

    return EA

#--------------------------
def update_RA(RA, SA, EA, K = 16.):
    '''
        compute the new rating of player A after playing a game.
        Input:
            RA: the current rating of player A, a float scalar value
            SA: the game result of player A, a float scalar value.
                if A wins in a game, SA = 1;if A loses, SA =0.
            EA: the expected probability of player A to win in the game, a float scalar between 0 and 1.
             K: k-factor, a contant number which controls how fast to correct the ratings
        Output:
            RA_new: the new rating of player A, a float scalar value
    '''
    RA_new = RA + K * (SA - EA)
    return RA_new


#--------------------------
def elo_rating(W, n_player, K= 16.):
    ''' 
        An implementation of Elo rating algorithm, which was used in facemash.
        Given a collection of game results W, compute the Elo rating scores of all the players.
        Input: 
                W: (wins) game results, a numpy matrix of shape (n_game,2), dtype as integers. If player i wins player j in the k-th game, W[k][0] = i, W[k][1] = j.
                n_player: the number of players to rate, an integer scalar.
                K: k-factor, a contant number which controls how fast to correct the ratings
        Output: 
                R: the Elo rating scores,  a python array of float values, such as [1000., 200., 500.], of length num_players
    '''

    # initialize the ratings of all players with 400
    R = n_player * [400.] 
   
    # for each game, update the ratings
    for (A, B) in W:
        # the game result: player A (win), player B (loss)
        # A is the index of player A, B is the index of player B

        # update player A's rating
        RA = R[A]
        RB = R[B]
        R[A] = update_RA(RA, 1, compute_EA(RA, RB), K)

        # update player B's rating
        R[B] = update_RA(RB, 0, compute_EA(RB, RA), K)

    return R 
