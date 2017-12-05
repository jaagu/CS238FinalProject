#https://github.com/seemethere/nba_py
from nba_py import team
from nba_py.constants import TEAMS
from datetime import datetime
import pandas as pd
teamToIndex = {
    'ATL': 0,
    'BOS': 1,
    'BKN': 2,  
    'CHA': 3,
    'CHI': 4,
    'CLE': 5,
    'DAL': 6,
    'DEN': 7,
    'DET': 8,
    'GSW': 9,
    'HOU': 10,
    'IND': 11,
    'LAC': 12,
    'LAL': 13,
    'MEM': 14,
    'MIA': 15,
    'MIL': 16,
    'MIN': 17,
    'NOP': 18,
    'NYK': 19,
    'OKC': 20,
    'ORL': 21,
    'PHI': 22,
    'PHX': 23,
    'POR': 24,
    'SAC': 25,
    'SAS': 26,
    'TOR': 27,
    'UTA': 28,
    'WAS': 29
}

#   Loads boxScores
def get_teamBoxScore(teamName, season):
    #Use nba_py to load data
    team_id = TEAMS[teamName]['id']
    df = team.TeamGameLogs(team_id, season).info()
    return df

def get_season(year):
    CURRENT_SEASON = str(year) + "-" + str(year + 1)[2:]
    return CURRENT_SEASON

#   Loads and adds games for teamName between startYear and endYear seasons to one table
def load_teamBoxScoresBetweenYears(teamName, startYear, endYear):
    df = get_teamBoxScore(teamName, get_season(startYear))
    for i in range(endYear-startYear):
        season = get_season(startYear+1+i)
        df = df.append(get_teamBoxScore(teamName, season), ignore_index=True)
    return df

