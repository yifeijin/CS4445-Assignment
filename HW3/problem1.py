import pandas as pd
# -------------------------------------------------------------------------
'''
    Problem 1: Sabermetrics 
    In this problem, you will implement a version of the baseball player ranking system.
    You could test the correctness of your code by typing `nosetests test1.py` in the terminal.
'''

#--------------------------
def batting_average(H, AB):
    '''
        compute the batting average of a player. 
        For more details, see here https://en.wikipedia.org/wiki/Batting_average.
        Input:
            H: the number of hits, an integer scalar. 
            AB: the number of "at bats",  an integer scalar
        Output:
            BA: the batting average of a player, a float scalar.
    '''
    BA = H / AB

    return BA


#--------------------------
def on_base_percentage(H, AB, BB, HBP, SF):
    '''
        compute the on base percentage of a player. 
        For more details, see here https://en.wikipedia.org/wiki/On-base_percentage.
        Input:
            H: the number of hits, an integer scalar. 
            AB: the number of "at bats",  an integer scalar
            BB: the number of bases on balls (walks),  an integer scalar
            HBP: the number of hit by pitch,  an integer scalar
            SF: the number of sacrifice fly,  an integer scalar
        Output:
            OBP: the on base percentage of a player, a float scalar.
    '''
    OBP = (H + BB + HBP) / (AB + BB + HBP + SF)

    return OBP 

#--------------------------
def slugging_percentage(H, _2B, _3B, HR, AB):
    '''
        compute the slugging percentage of a player. 
        For more details, see here https://en.wikipedia.org/wiki/Slugging_percentage.
        Input:
            H: the number of hits, an integer scalar. 
            _2B: the number of 2nd base,  an integer scalar (note: python variable names cannot start with a number, so _ is added)
            _3B: the number of 3rd base,  an integer scalar
            HR: the number of home runs,  an integer scalar
            AB: the number of at bats,  an integer scalar
        Output:
            SLG: the slugging percentage of a player, a float scalar.
    '''
    SLG = (H + _2B + 2 * _3B + 3 * HR) / AB

    return SLG 


#--------------------------
def runs_created(H, _2B, _3B, HR, BB, AB):
    '''
        compute the expected runs created by a team based upon Bill James' runs created formula. 
        For more details, see here https://en.wikipedia.org/wiki/Runs_created.
        Input:
            H: the number of hits, an integer scalar. 
            _2B: the number of 2nd base,  an integer scalar (note: python variable names cannot start with a number, so _ is added)
            _3B: the number of 3rd base,  an integer scalar
            HR: the number of home runs,  an integer scalar
            BB: the number of bases on balls (walks),  an integer scalar
            AB: the number of at bats,  an integer scalar
        Output:
            RC: the expected runs created/scored by a team, a float scalar.
    '''
    RC = (H + BB) * (H + _2B + 2 * _3B + 3 * HR) / (AB + BB)

    return RC 


#--------------------------
def win_ratio(RC, RA):
    '''
        compute the expected wining ratio of a team based upon Bill James' Pythagorean expectation. 
        For more details, see here https://en.wikipedia.org/wiki/Pythagorean_expectation.
        Input:
            RC: the number of runs created/scored, an integer scalar. 
            RA: the number of runs allowed,  an integer scalar
        Output:
            WR: the projected winning ratio of a team, a float scalar.
    '''
    WR = RC ** 2 / (RC ** 2 + RA ** 2)

    return WR 

#--------------------------
def eval_player(ID='hattesc01'):
    '''
        compute the BA, OBP and SLG of a player (ID) from the data in "Batting.csv" (year 2001). 
        This dataset is downloaded from Lahman's baseball database (http://www.seanlahman.com/baseball-archive/statistics/)
        Input:
            ID: the player ID of a player, a string.  For example, "hattesc01" is for Scott Hatteberg. You can find the player names and IDs in People.csv.
        Output:
            BA: the batting average of a player, a float scalar. 
            OBP: the on base percentage of a player, a float scalar.
            SLG: the slugging percentage of a player, a float scalar.
        Hint: you could use pandas package to load csv file and search player. Here is a tutorial: http://pandas.pydata.org/pandas-docs/stable/10min.html
    '''
    # read Batting.csv
    df = pd.read_csv("Batting.csv")

    # search player by player ID
    player = df[df.playerID == ID].iloc[0]
    H = player.H
    AB = player.AB
    BB = player.BB
    HBP = player.HBP
    SF = player.SF
    _2B = player['2B']
    _3B = player['3B']
    HR = player.HR

    BA = batting_average(H, AB)
    OBP = on_base_percentage(H, AB, BB, HBP, SF)
    SLG = slugging_percentage(H, _2B, _3B, HR, AB)

    return BA, OBP, SLG

#--------------------------
def salary(ID='hattesc01'):
    '''
        find the salary of a player in "Salaries.csv" (year 2002). 
        Input:
            ID: the player ID of a player, a string.  For example, "hattesc01" is for Scott Hatteberg. You can find the player names and IDs in People.csv.
        Output:
            S: the salary of a player in year 2002, a float scalar.
        Hint: you could use pandas package to load csv file and search player. Here is a tutorial: http://pandas.pydata.org/pandas-docs/stable/10min.html
    '''


    # read Batting.csv
    df = pd.read_csv("Salaries.csv")

    # search player by player ID
    S = df[df.playerID == ID].iloc[0].salary

    return S 



