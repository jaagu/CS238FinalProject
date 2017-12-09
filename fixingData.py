import pandas as pd
import numpy as np

def main():
    df = pd.read_csv("data127Night.csv")
    matchups = df["MATCHUP"].values
    firstteamPts = df["PTS"].values
    i = 0
    for ptsList in df["points"]:
        splitString = ptsList[1:-1].split()
        pts = []
        for j in range(len(splitString)):
            pts.append(float(splitString[j]))
        matchup = matchups[i]
        home = 0
        if '@' in matchup:
            home = 0
        else:
            home = 1
        pts = np.array(pts)
        indices = [k for k, e in enumerate(pts) if e != 0]
        newPts = pts.copy()
        if home:
            if (pts[indices[0]] != firstteamPts[i]):
                newPts[indices[0]] = pts[indices[1]]
                newPts[indices[1]] = pts[indices[0]]
        else:
            if (pts[indices[1]] != firstteamPts[i]):
                newPts[indices[0]] = pts[indices[1]]
                newPts[indices[1]] = pts[indices[0]]
        df = df.set_value(i, "points", newPts)
        i+=1
    df.to_csv("FIXED_DATA.csv")
    return

if __name__ == '__main__':
    main()
