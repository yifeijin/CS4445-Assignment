from problem1 import *
import numpy as np
import sys

'''
    Unit test 1:
    This file includes unit tests for problem1.py.
    You could test the correctness of your code by typing `nosetests -v test1.py` in the terminal.
'''

#-------------------------------------------------------------------------
def test_python_version():
    ''' ----------- Problem 1 (40 points in total)---------------------'''
    assert sys.version_info[0]==3 # require python 3 (instead of python 2)

#-------------------------------------------------------------------------
def test_R_play():
    '''(5 points) Player Random play()'''
    p = PlayerRandom()
    s=np.array([[ 0, 1, 1],
                [ 1, 0,-1],
                [ 1, 1, 0]])

    s_=np.array([[ 0, 1, 1],
                [ 1, 0,-1],
                [ 1, 1, 0]])
    count=np.zeros(3)
    for _ in range(100):
        r,c = p.play(s)
        assert s[r,c]==0 
        assert r==c 
        assert r>-1 and r<3
        assert np.allclose(s,s_)
        count[c]+=1
    assert count[0]>20
    assert count[1]>20
    assert count[2]>20
    
    s=np.array([[ 1, 1, 0],
                [ 1, 0,-1],
                [ 0, 1, 1]])

    for _ in range(100):
        r,c = p.play(s)
        assert s[r,c]==0 
        assert r==2-c 
        assert r>-1 and r<3
 

#-------------------------------------------------------------------------
def test_T_play_x():
    '''(2 points) TicTacToe play_x()'''
    g = TicTacToe()
    g.play_x(0,0) 
    assert np.allclose(g.s[0,0],1)
    assert np.allclose(g.s.sum(),1)
    g.play_x(2,1) 
    assert np.allclose(g.s[2,1],1)
    assert np.allclose(g.s.sum(),2)


#-------------------------------------------------------------------------
def test_T_play_o():
    '''(2 points) TicTacToe play_o()'''
    g = TicTacToe()
    g.play_o(0,0) 
    assert np.allclose(g.s[0,0],-1)
    assert np.allclose(g.s.sum(),-1)
    g.play_o(2,2) 
    assert np.allclose(g.s[2,2],-1)
    assert np.allclose(g.s.sum(),-2)

#-------------------------------------------------------------------------
def test_T_avail_moves():
    '''(2 points) TicTacToe available_moves()'''
    g = TicTacToe()

    s=np.array([[ 0, 1, 1],
                [ 1, 0,-1],
                [ 0, 1, 0]])

    m=g.avail_moves(s)
    assert m[0][0]==0 
    assert m[0][1]==0 
    assert m[1][0]==1 
    assert m[1][1]==1 
    assert m[2][0]==2 
    assert m[2][1]==0 
    assert m[3][0]==2 
    assert m[3][1]==2 


#-------------------------------------------------------------------------
def test_T_check():
    '''(5 points) TicTacToe check()'''
    g = TicTacToe()
    e = g.check(g.s)
    assert e is None 
    g.play_x(0,0) 
    g.play_x(0,1) 
    g.play_x(0,2) 
    e = g.check(g.s)
    assert e == 1 
    
    g = TicTacToe()
    g.play_o(0,0) 
    g.play_o(1,0) 
    g.play_o(2,0) 
    e = g.check(g.s)
    assert e == -1
    
    g = TicTacToe()
    g.play_o(0,2) 
    g.play_o(1,1) 
    g.play_o(2,0) 
    e = g.check(g.s)
    assert e == -1
    
    g = TicTacToe()
    g.play_o(2,2) 
    g.play_o(1,1) 
    g.play_o(0,0) 
    e = g.check(g.s)
    assert e == -1

    g = TicTacToe()
    g.s=np.array([[-1, 1,-1],
                  [-1, 1,-1],
                  [ 1,-1, 1]])
    e = g.check(g.s)
    assert e == 0 

    g.s=np.array([[-1, 0,-1],
                  [-1, 1,-1],
                  [ 1,-1, 1]])
    e = g.check(g.s)
    assert e is None 


#-------------------------------------------------------------------------
def test_T_game():
    '''(3 points) TicTacToe game()'''

    p1 = PlayerRandom()
    p2 = PlayerRandom()
    w =0  
    for i in range(100):
        g = TicTacToe()
        g.s=np.array([[ 0,-1, 1],
                      [-1, 1, 0],
                      [-1, 1,-1]])
        e = g.game(p1,p2)
        w+=e
    print(w)
    assert w<0
    assert w<-30
    assert w>-70

    class xplayer:
        def play(self,s,x=1):
            assert s[1,1] == 1
            assert s[2,2] ==-1
            assert x ==1 # because this is an "X" player
            r,c=np.where(s==0)
            return r[0],c[0]
    class oplayer:
        def play(self,s,x=1):
            assert s[1,1] == 1
            assert s[2,2] ==-1
            assert x ==-1 # because this is an "O" player
            r,c=np.where(s==0)
            return r[0],c[0]

    g = TicTacToe()

    p1 = xplayer()
    p2 = oplayer()
    
    g.s=np.array([[ 0, 0, 0],
                  [ 0, 1, 0],
                  [ 0, 0,-1]])
     
    e = g.game(p1,p2)

    class test3:
        def play(self,s,x=1):
            r,c=np.where(s==0)
            return r[0],c[0]

    g = TicTacToe()

    p1 = test3()
    e = g.game(p1,p1)
    print(g.s)
    s=np.array([[ 1,-1, 1],
                [-1, 1,-1],
                [ 1, 0, 0]])
    assert np.allclose(g.s, s)
    assert e==1

    g = TicTacToe()
    s=np.array([[ 1,-1, 1],
                [-1,-1,-1],
                [-1, 1, 1]])
    g.s=s
    e = g.game(p1,p1)
    assert e==-1


#-------------------------------------------------------------------------
def test_expand():
    '''(5 points) expand'''
    s=np.array([[ 0,-1,-1],
                [ 0, 1, 1],
                [-1, 1,-1]])
    n = Node(s,x=1)
    n.expand()
    assert len(n.c) ==2 
    assert n.x == 1
    s_=np.array([[ 0,-1,-1],
                 [ 0, 1, 1],
                 [-1, 1,-1]])
    assert np.allclose(n.s,s_)
    for c in n.c:
        assert c.x==-1

    s=np.array([[ 0,-1,-1],
                [ 1, 1, 1],
                [-1, 1,-1]])
    c = False
    for x in n.c:
        if np.allclose(x.s,s):
            c=True
    assert c

    s=np.array([[ 1,-1,-1],
                [ 0, 1, 1],
                [-1, 1,-1]])
    c = False
    for x in n.c:
        if np.allclose(x.s,s):
            c=True
    assert c



    s=np.array([[ 0,-1,-1],
                [-1, 1, 1],
                [-1, 1,-1]])
    n = Node(s,-1)
    n.expand()
    assert n.x==-1
    assert len(n.c) ==1
    assert n.c[0].x==1


    s=np.array([[ 0,-1,-1],
                [ 1, 1, 1],
                [-1, 1,-1]])
    n = Node(s)
    n.expand()
    assert len(n.c) ==0


#-------------------------------------------------------------------------
def test_build_tree():
    '''(2 points) build_tree'''

    s=np.array([[ 0,-1, 1],
                [ 0, 1, 1],
                [-1, 1,-1]])
    n = Node(s, x=-1)
    n.build_tree()

    assert len(n.c) ==2 
    assert n.x == -1
    
    s1=np.array([[-1,-1, 1],
                 [ 0, 1, 1],
                 [-1, 1,-1]])
    s2=np.array([[ 0,-1, 1],
                 [-1, 1, 1],
                 [-1, 1,-1]])
    for c in n.c:
        assert c.x==1
        assert len(c.c) ==1
        if np.allclose(c.s,s1):
            assert c.m == (0,0)
        if np.allclose(c.s,s2):
            assert c.m == (1,0)


    s=np.array([[ 0,-1,-1],
                [ 0,-1,-1],
                [ 0, 1, 1]])
    n = Node(s,x=-1)
    n.build_tree()

    assert len(n.c) ==3 
    assert n.x==-1
    
    for c in n.c:
        assert c.x==1
        assert len(c.c) ==0

#-------------------------------------------------------------------------
def test_M_compute_v():
    '''(5 points) Player MiniMax compute_v()'''
    s=np.array([[ 1, 0, 0],
                [ 0, 1, 0],
                [ 0,-1, 1]])
    n = Node(s, x=-1)
    n.build_tree()
    n.compute_v() # X player won the game
    assert  n.v== 1

    n = Node(s, x= 1)
    n.build_tree()
    n.compute_v() # X player won the game
    assert  n.v== 1

    s=np.array([[-1, 0, 0],
                [ 0,-1, 0],
                [ 0, 1,-1]])
    n = Node(s, x= 1)
    n.build_tree()
    n.compute_v() # O player won the game
    assert  n.v== -1

    n = Node(s, x=-1)
    n.build_tree()
    n.compute_v() # X player won the game
    assert  n.v==-1

    s=np.array([[-1, 1,-1],
                [-1, 1,-1],
                [ 1,-1, 1]])
    n = Node(s, x=-1)
    n.build_tree()
    n.compute_v() # tie 
    assert  n.v== 0

    s=np.array([[-1, 1,-1],
                [-1, 1, 1],
                [ 0,-1, 1]])
    n = Node(s, x= 1)
    n.build_tree()
    n.compute_v() # a tie after one move 
    assert  n.v== 0


    s=np.array([[-1,-1, 1],
                [-1, 1,-1],
                [ 0,-1, 1]])
    n = Node(s, x= 1)
    n.build_tree()
    n.compute_v() # X player wins after one move 
    assert  n.v==1

    n = Node(s, x=-1)
    n.build_tree()
    n.compute_v() # O player wins after one move 
    assert  n.v==-1

    s=np.array([[ 0, 1,-1],
                [-1,-1, 1],
                [ 0,-1, 1]])
    n = Node(s, x= 1)
    n.build_tree()
    n.compute_v() # a tie no matter which move 
    assert  n.v==0


    s=np.array([[ 0, 1, 1],
                [-1, 1,-1],
                [-1,-1, 0]])
    n = Node(s, x= 1)
    n.build_tree()
    n.compute_v() # the best move leads to win (X player won)
    assert  n.v==1

    s=np.array([[ 1, 0, 0],
                [ 0, 0,-1],
                [ 0, 0,-1]])
    n = Node(s, x= 1)
    n.build_tree()
    n.compute_v() # the best moves lead to win (X player won)
    assert  n.v==1

    c=n.c[0]
    assert c.x==-1 

    s=np.array([[ 0, 0,-1],
                [ 1, 0, 0],
                [ 1, 0, 0]])
    n = Node(s, x=-1)
    n.build_tree()
    n.compute_v() # the best moves lead to win (O player won)
    assert  n.v==-1


#-------------------------------------------------------------------------
def test_M_play():
    '''(5 points) Player MiniMax play()'''
    p = PlayerMiniMax()
    s=np.array([[-1, 1,-1],
                [-1, 1,-1],
                [ 0,-1, 1]])
    r, c = p.play(s)
    assert np.allclose(s,[[-1, 1,-1], [-1, 1,-1], [ 0,-1, 1]])
    assert r==2  
    assert c==0


    p = PlayerMiniMax()
    s=np.array([[-1,-1, 1],
                [-1, 1,-1],
                [ 0,-1, 1]])
    r, c = p.play(s)
    assert r==2
    assert c==0  

    p = PlayerMiniMax()
    s=np.array([[ 0,-1, 1],
                [-1, 1,-1],
                [ 0,-1,-1]])
    r, c = p.play(s)
    assert r==2  
    assert c==0  

    p = PlayerMiniMax()
    s=np.array([[ 0, 1,-1],
                [-1,-1, 1],
                [ 0,-1, 1]])
    r, c = p.play(s)
    assert r==2  
    assert c==0  

    p = PlayerMiniMax()
    s=np.array([[ 0, 1, 1],
                [-1, 1,-1],
                [-1,-1, 1]])
    r, c = p.play(s)
    assert r==0  
    assert c==0  

    p = PlayerMiniMax()
    s=np.array([[ 0,-1, 1],
                [-1, 1,-1],
                [-1, 1, 0]])
    r, c = p.play(s)
    assert r==0  
    assert c==0  

    p = PlayerMiniMax()
    s=np.array([[ 0, 0,-1],
                [ 1, 0, 0],
                [ 1, 0, 0]])

    r, c = p.play(s,x=-1) 
    assert r==0  
    assert c==0  
    s=np.array([[-1, 0,-1],
                [ 1, 0, 0],
                [ 1, 1, 0]]) # if the other player choose (2,1)

    r, c = p.play(s,x=-1) # we choose move 
    assert r==0  
    assert c==1  

#-------------------------------------------------------------------------
def test_players():
    '''(2 points) random vs Minimax'''

    p2 = PlayerRandom()
    w=0
    for i in range(100):
        g = TicTacToe()
        g.s=np.array([[ 0,-1, 1],
                      [-1, 1,-1],
                      [ 0,-1,-1]])
        p1 = PlayerMiniMax()
        e = g.game(p1,p2)
        w += e
    assert w==100

    w=0
    for i in range(100):
        g = TicTacToe()
        g.s=np.array([[ 0,-1, 1],
                      [-1, 1,-1],
                      [-1, 1, 0]])
        p1 = PlayerMiniMax()
        e = g.game(p1,p2)
        w += e
    assert w==0


    w=0
    for i in range(100):
        g = TicTacToe()
        g.s=np.array([[ 0, 0, 1],
                      [ 0,-1, 0],
                      [ 1,-1, 0]])
        p1 = PlayerMiniMax()
        e = g.game(p1,p2)
        w += e
    assert np.abs(w-87)<10

#-------------------------------------------------------------------------
def test_players2():
    '''(2 points) Minimax vs Minimax'''
    g = TicTacToe()
    g.s=np.array([[ 0, 0, 1],
                  [ 0,-1, 0],
                  [ 1,-1, 0]])
    p1 = PlayerMiniMax()
    p2 = PlayerMiniMax()
    e = g.game(p1,p2)
    assert e==0

    g = TicTacToe()
    g.s=np.array([[ 0, 0, 0],
                  [ 0,-1, 0],
                  [ 1, 0, 0]])
    p1 = PlayerMiniMax()
    p2 = PlayerMiniMax()
    e = g.game(p1,p2)
    assert e==0

    g = TicTacToe()
    g.s=np.array([[ 0, 0, 0],
                  [ 0, 0, 0],
                  [ 1,-1, 0]])
    p1 = PlayerMiniMax()
    p2 = PlayerMiniMax()
    e = g.game(p1,p2)
    assert e==1

    g = TicTacToe()
    g.s=np.array([[ 0, 0, 0],
                  [ 0, 1, 0],
                  [ 0,-1, 0]])
    p1 = PlayerMiniMax()
    p2 = PlayerMiniMax()
    e = g.game(p1,p2)
    assert e==1

    g = TicTacToe()
    g.s=np.array([[ 0, 0, 0],
                  [ 0, 1, 0],
                  [-1, 0, 0]])
    p1 = PlayerMiniMax()
    p2 = PlayerMiniMax()
    e = g.game(p1,p2)
    assert e==0

    # the following test run a complete game, but it may take 3 minutes to run
    #g = TicTacToe()
    #p1 = PlayerMiniMax()
    #p2 = PlayerMiniMax()
    #e = g.game(p1,p2)
    #assert e==0


