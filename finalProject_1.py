import getData
from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


ofeatures = ['homeAway', 'L', 'W']

def add_homeaway_teams(df):
    home = []
    tpArray = []
    dateweightArray = []
    dates=df['GAME_DATE'].values
    i = 0
    for matchup in df['MATCHUP']:
        teamsPlaying = np.zeros(len(getData.teamToIndex))
        if '@' in matchup:
            home.append(0)
        else:
            home.append(1)
        teams = matchup.split(" ")
        t1Index = getData.teamToIndex[teams[0]]
        t2Index = getData.teamToIndex[teams[2]]
        teamsPlaying[t1Index] = 1
        teamsPlaying[t2Index] = 1
        tpArray.append(teamsPlaying)

        #add date weight
        date = dates[i].split(" ")
        dateweightArray.append(0.9**(2017-int(date[2])+1))
        i += 1
        
    ha = pd.Series(home)
    tp = pd.Series(tpArray)
    dw = pd.Series(dateweightArray)
    df = df.assign(homeAway = ha.values)
    df = df.assign(playingTeams = tp.values)
    df = df.assign(dateWeight = dw.values)
        
    return df

def winLossToBool(truth):
    for i in range(len(truth)):
        if truth[i] == 'W':
            truth[i] = 1
        else:
            truth[i] = 0
    return truth

def addOtherFeaturesToPlayingTeams(dataframe, feature, additionalfeatures):
    features = dataframe[feature].values.tolist()
    otherfeatures = dataframe[additionalfeatures].values
    for i in range(len(features)):
        featurelist = features[i].tolist()
        for j in range(len(otherfeatures[i])):
            if(otherfeatures[i][j] != 'nan'):
                featurelist.append(otherfeatures[i][j])
            else:
                print('hi')
                featurelist.append(0)
        features[i] = featurelist
    return features

def trainSVM(train):
    clf = svm.SVC(kernel='rbf')
    features = addOtherFeaturesToPlayingTeams(train,'playingTeams',ofeatures)
   # print(features)
    truth = winLossToBool(train['WL'].values.tolist())
    clf.fit(features,truth)
    return clf

def testSVM(clf, test):
    truth = winLossToBool(test['WL'].values.tolist())
    features = addOtherFeaturesToPlayingTeams(test,'playingTeams',ofeatures)
    #print(features)
    correct = 0
    total = len(test)
    for t in range(len(test)):
        if(truth[t] == 'nan'):
            print('nan')
        if(truth[t] == clf.predict([features[t]])):
           correct+=1
    return(correct/float(total))

def main():
    X = []
    Y = []
##    #load data from the internet
##    df = getData.load_teamBoxScoresBetweenYears('DEN',2013,2017)
##    totalAccuracy = 0
##    for _ in range(100):
##        #get train, test
##        train, test = train_test_split(df, test_size=0.2)
##        #modify data
##        train = add_homeaway_teams(train)
##        test = add_homeaway_teams(test)
##     #   print(train)
##        #train svm
##        clf = trainSVM(train)
##        #test svm
##        totalAccuracy += testSVM(clf, test)
##    averageError = totalAccuracy/float(100)
##    print(averageError)
    i = 1
    teams = []
    for team in getData.teamToIndex.keys():
        X.append(i)
        teams.append(team)
        i += 1
        print(team)
        #load data from the internet
        df = getData.load_teamBoxScoresBetweenYears(team,2013,2017)
        totalAccuracy = 0
        for _ in range(100):
            #get train, test
            train, test = train_test_split(df, test_size=0.2)
            #modify data
            train = add_homeaway_teams(train)
            test = add_homeaway_teams(test)
            #train svm
            clf = trainSVM(train)
            #test svm
            totalAccuracy += testSVM(clf, test)
        averageError = totalAccuracy/float(100)
        print(averageError)
        Y.append(averageError)
   
    plt.plot(X,Y, marker = 'o')
    plt.xticks(X, teams, rotation=45, fontsize = 10)
    plt.ylim(ymin=0.3, ymax=1)

    plt.plot([0, 31], [0.5,0.5], color = 'red')
    plt.grid(True)
    plt.show()
    return

if __name__ == '__main__':
    main()
