import pandas as pd
import numpy as np

dfPlayers = pd.read_csv('data.csv')
# data yg di pake 'Name', 'Age', 'Overall', 'Potential',
dfPlayers = dfPlayers.drop(columns = ['ID','Photo', 'Nationality', 'Flag',
            'Club', 'Club Logo', 'Value', 'Wage', 'Special',
            'Preferred Foot', 'International Reputation', 'Weak Foot',
            'Skill Moves', 'Work Rate', 'Body Type', 'Real Face', 'Position',
            'Jersey Number', 'Joined', 'Loaned From', 'Contract Valid Until',
            'Height', 'Weight', 'LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW',
            'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM',
            'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB', 'Crossing',
            'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',
            'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',
            'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',
            'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',
            'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
            'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',
            'GKKicking', 'GKPositioning', 'GKReflexes', 'Release Clause'])
dfPlayers = dfPlayers.drop(dfPlayers.columns[0], axis=1)
# print(dfPlayers .head())
# print(dfPlayers .info)
# print(dfPlayers.columns)

# Usia (Age) <= 25 tahun,
# Skill umum (Overall) >= 80 point, dan
# Potensi (Potential) >= 80 point.
targetPlayers = dfPlayers[dfPlayers['Age']<=25]
targetPlayers = targetPlayers[targetPlayers['Overall']>=80]
targetPlayers = targetPlayers[targetPlayers['Overall']>=80]
targetPlayers['Stats'] = 1
# print(targetPlayers.head())
nonTargetPlayers = dfPlayers.drop(targetPlayers.index)
nonTargetPlayers['Stats'] = 0
# print(nonTargetPlayers.head())

Train = pd.concat([targetPlayers,nonTargetPlayers], axis = 0).drop(columns = 'Name')
# print(len(Train))
Stats = ['nonTargetPlayers','targetPlayers']

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

def x():
    x = round(len(Train) ** 0.5)
    if x % 2 == 0:
        return x + 1
    else:
        return x
# print(cross_val_score(DecisionTreeClassifier(),Train[['Age','Overall', 'Potential']], Train['Stats'], cv=5))
# print(cross_val_score(KNeighborsClassifier(n_neighbors=x()),Train[['Age','Overall', 'Potential']], Train['Stats'], cv=5))
# print(cross_val_score(RandomForestClassifier(n_estimators=100),Train[['Age','Overall', 'Potential']], Train['Stats'], cv=5))

# [0.7188358 1.        1.        1.        1.       ] DTC
# [0.80669962 0.99972543 0.99697968 0.99423235 0.99093407] KNN
# [0.7188358  1.         1.         1.         0.99642857] RFC

DTC = np.mean(np.array([0.7188358,1.,1.,1.,1.]))
# print(DTC)
KNN = np.mean(np.array([0.80669962,0.99972543,0.99697968,0.99423235,0.99093407]))
# print(KNN)
RFC = np.mean(np.array([0.7188358,1.,1.,1.,0.99642857]))
# print(RFC)
# KKN tertinggi
KNN = KNeighborsClassifier(n_neighbors=x())
KNN.fit(Train[['Age','Overall', 'Potential']], Train['Stats'])

df_IDN = pd.DataFrame(np.array([
                        ['Andik Vermansyah','Madura United FC',27,87,90],
                        ['Awan Setho Raharjo','Bhayangkara FC',22,75,83],
                        ['Bambang Pamungkas','Persija Jakarta',38,85,75],
                        ['Cristian Gonzales','PSS Sleman',43,90,85],
                        ['Egy Maulana Vikri','Lechia Gdańsk',8,88,90],
                        ['Evan Dimas','Barito Putera',24,85,87],
                        ['Febri Hariyadi','Persib Bandung',23,77,80],
                        ['Hansamu Yama Pranata','Persebaya Surabaya',24,82,85],
                        ['Septian David Maulana','PSIS Semarang',22,83,80],
                        ['Stefano Lilipaly','Bali United',29,88,86]]),columns = ['Name','Club','Age','Overall','Potential'])
df_IDN['Nationality'] = 'Indonesia'
# print(df_IDN)

Recruit_Player = ['nonTargetPlayers','targetPlayers']

df_IDN['Prediksi'] = KNN.predict(df_IDN[['Age','Overall', 'Potential']])
df_IDN['Prediksi'] = df_IDN['Prediksi'].apply(lambda i: Recruit_Player[i])
print(df_IDN)