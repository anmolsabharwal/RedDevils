#!/usr/bin/env python
# coding: utf-8

# ## Fantasy Premier League Points Predictor 
# The team consists of:
# 1. Aashish Harneja - Subject Matter Expert
# 2. Sagar Patnaik - Subject Matter Expert
# 3. Anmol Sabharwal - Data Engineer
# 4. Raghav Bhutra - Data Engineer
# 
# The project aims at predicting which player should be drafted in a team based on the predicted points scored by him in the upcoming week. The data is scraped from Fantasy Premier League's website after each game week. We utilize N-1 weeks worth of data. Extra features are also created to perform further analysis on how players have been performing over the season and notice trends in performances. 
# 
# For future steps we plan to move to a regression based model in order to provide specific results.
# 
# **Assumptions**- Even though we can achieve high accuracy with the model, we cannot neglect the fact that this is a sport and the variation is always high. We cannot account for human behaviour for example a player having a bad day because of personal reasons. We do not take into account the player's relationship with the manager, we mention this because manager has the right t0 bench or not play a particular player. We also do not account for injuries **yet**
# 

# **Importing Main Libraries**

# In[45]:


#importing Important libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sqlite3
import datetime
import sqlite3
 
get_ipython().run_line_magic('matplotlib', 'inline')


# #### Loading Data
# 
# **After importing the necessary libraries, we will establish a connection with the local sqllite database. The name of database is 'fpl' and it contains a single table by the name 'gw'. Once we establish the connection we can execute queries just like we do in SQL environment**

# In[46]:


#connection to .db file
conn = sqlite3.connect("/Users/anmolsabharwal/Desktop/fpl.db")
cur = conn.cursor()
gws = pd.read_sql_query("SELECT * FROM gw", conn)


# In[47]:


gws.columns


# #### Data Preprocessing 
# **For the ease of use and consistency, convert the columns to appropriate data type.**

# In[48]:


#converting the datatypes of columns
gws=gws.astype({'name':'category','assists':int, 'bonus':int, 'bps':int, 'clean_sheets':int,
       'creativity':float,'element':int,'goals_conceded':int,
       'goals_scored':int, 'ict_index':float, 'influence':float,'minutes':int, 'opponent_team':int, 'own_goals':int,'penalties_missed':int, 'penalties_saved':int,
       'red_cards':int, 'round':int, 'saves':int, 'selected':int,
        'threat':float,
       'total_points':int, 'transfers_balance':int, 'transfers_in':int, 'transfers_out':int,
       'value':int, 'yellow_cards':int,'gw':int})


# **Once we have the columns with the correct data type we check for any NULL values present in the dataset. To make things easier, we use the Seaborn library to plot the heatmap. Since it returns a complete black heatmap, it means there are no missing values**

# In[49]:


# cheking for null values
sns.set_style('dark')
sns.heatmap(gws.isnull(),yticklabels=False,cbar=False,cmap='inferno')


# **The names provided in the dataset are strings, just to make things easier we are going to assign unique ID's based on names.**

# In[50]:


#assigning id to each specific player
gws = gws.assign(id=(gws['name']).astype('category').cat.codes)


# **To reduce noise in the data, we will remove all those players who havent played a match and we do this by using a filter on 'minutes' column. We chose the minutes column instead of points because a player can have 0 points even after playing for 90 minutes**

# In[51]:


gws= gws[gws['minutes'] > 0] 


# **This function decides the difficulty of a particular fixture based on the position of a team in a table. Teams in the top half of the table will have highest fixture difficulty ratio while the teams in the bottom half will have lower fixture difficulty ratio.**

# In[52]:


# a function to decide the fixture difficulty ratio based on the position of the opponent team
def fd(x):
  x=int(x)
  if x<=6:
    return '3'
  elif x>6 and x<=14:
    return '1'
  else:
    return '2'


# In[53]:


gws.rename(columns={'opponent_team':'fdr'},inplace=True)


# In[54]:


gws['fdr']=gws['fdr'].apply(fd).astype(float)


# **The matches usually take place at different times of the day. We first extracted the hour from the datetime and then divided it into 3 categories to see if it has any effect on the perfomance of a player**

# In[55]:


gws['kickoff_time']=pd.to_datetime(gws['kickoff_time'])


# In[56]:


gws['kickoff_time'] = gws.kickoff_time.dt.hour


# In[57]:


#divind the kickoff time into 3 categories 
def time(x):
  x=int(x)
  if x>8 and x<=12:
    return '0'
  elif x>12 and x<=18:
    return '1'
  else:
    return '2'


# In[58]:


gws['kickoff_time']=gws['kickoff_time'].apply(time).astype(float)
gws.head()


# **While exploring different variables that affect the performance of a player we wanted to focus on how he performs at home matches and away matches. In order to that, we change the initial boolean 'True' and 'False' values to 'Home' and 'Away'. After that, we performed one-hot encoding to get different columns for both Home and Away.**

# In[59]:


# setting up the 'was_home' column to create binary attributes
def was_home(a):
  a=a.upper()
  if a=="FALSE":
    return str('Away')
  else:
    return str('Home')


# In[60]:


gws['home']=gws['was_home'].apply(was_home)


# **Feature engineered variables based on Home and away performances of a player for further analysis**

# In[61]:


gws['Away Points'] = np.where(gws['home'] == 'Home', gws['total_points'] * 0,gws['total_points'])
gws['Home Points'] = np.where(gws['home'] == 'Away', gws['total_points'] * 0,gws['total_points'])

gws['Away Influence'] = np.where(gws['home'] == 'Home', gws['influence'] * 0,gws['influence'])
gws['Home Influence'] = np.where(gws['home'] == 'Away', gws['influence'] * 0,gws['influence'])

gws['Away Goals'] = np.where(gws['home'] == 'Home', gws['goals_scored'] * 0,gws['goals_scored'])
gws['Home Goals'] = np.where(gws['home'] == 'Away', gws['goals_scored'] * 0,gws['goals_scored'])

gws['Away Saves'] = np.where(gws['home'] == 'Home', gws['saves'] * 0,gws['saves'])
gws['Home Saves'] = np.where(gws['home'] == 'Away', gws['saves'] * 0,gws['saves'])

gws['Away Creativity'] = np.where(gws['home'] == 'Home', gws['creativity'] * 0,gws['creativity'])
gws['Home Creativity'] = np.where(gws['home'] == 'Away', gws['creativity'] * 0,gws['creativity'])

gws['Away ICT'] = np.where(gws['home'] == 'Home', gws['ict_index'] * 0,gws['ict_index'])
gws['Home ICT'] = np.where(gws['home'] == 'Away', gws['ict_index'] * 0,gws['ict_index'])

gws['Away Bonus'] = np.where(gws['home'] == 'Home', gws['bonus'] * 0,gws['bonus'])
gws['Home Bonus'] = np.where(gws['home'] == 'Away', gws['bonus'] * 0,gws['bonus'])

gws['Away Clean Sheets'] = np.where(gws['home'] == 'Home', gws['clean_sheets'] * 0,gws['clean_sheets'])
gws['Home Clean Sheets'] = np.where(gws['home'] == 'Away', gws['clean_sheets'] * 0,gws['clean_sheets'])

gws['Away Penalties Saved'] = np.where(gws['home'] == 'Home', gws['penalties_saved'] * 0,gws['penalties_saved'])
gws['Home Penalties Saved'] = np.where(gws['home'] == 'Away', gws['penalties_saved'] * 0,gws['penalties_saved'])

gws['Away Goals Conceded'] = np.where(gws['home'] == 'Home', gws['goals_conceded'] * 0,gws['goals_conceded'])
gws['Home Goals Conceded'] = np.where(gws['home'] == 'Away', gws['goals_conceded'] * 0,gws['goals_conceded'])

gws['Away Assists'] = np.where(gws['home'] == 'Home', gws['assists'] * 0,gws['assists'])
gws['Home Assists'] = np.where(gws['home'] == 'Away', gws['assists'] * 0,gws['assists'])

gws.head()


# **Dropping columns that are not required**
# 

# In[62]:


#dropping columns that are not required
gws.drop(['element','fixture','round','transfers_balance','transfers_in','transfers_out','team_a_score','was_home',	'team_h_score','round'],axis=1, inplace=True)


# **The value of a player is the most important aspect of the game apart from the points. We are only given a budget on 100Mn to buy 15 players. The values in the original dataset were on a different scale, in order for it to makes sense we have adjusted the scale.**

# In[63]:


gws['value']=gws['value']*0.1


# **The algorithm makes a decision of a player being drafted or not purely on the basis of points for now. In the original dataset, we were given points in order for it to be converted into a classification problem we divided the points into two categories-<br></br>
# 1)Less than 7(Player will not be drafted)<br></br>
# 2)Greater than 7(Player will be drafted)**

# In[64]:


# a fucntion to put the points into the bins. This is where the business problem is convereted from a regression to a classification problem
def points(x):
  if x<8:
    return 0
  else:
    return 1


# In[65]:


gws['points']=gws['total_points'].apply(points)
gws1=gws.copy()
gws1.head()


# **Distribution of each class of the dependent variable. This clearly shows there is a class imbalance which is obvious because we are trying to separate and predict exceptional performances**

# In[66]:


sns.countplot(gws1['points'])


# **We need historical data to create variables so we try and make the algorithm understand how the player has been performing over the past few weeks to notice a pattern. This is called lags where the value lags for a week and is projected onto next week**
# 

# In[67]:


#creating lags for different attributes

gws1['Last_Week_Points'] = gws1.groupby(['id'])['total_points'].shift().astype(float)
gws1['Last_Week_Goals'] = gws1.groupby(['id'])['goals_scored'].shift().astype(float)
gws1['Last_Week_ICT'] = gws1.groupby(['id'])['ict_index'].shift().astype(float)
gws1['Last_Week_BPS'] = gws1.groupby(['id'])['bps'].shift().astype(float)
gws1['Last_Week_Assists'] = gws1.groupby(['id'])['assists'].shift().astype(float)
gws1['Last_Week_Bonus'] = gws1.groupby(['id'])['bonus'].shift().astype(float)
gws1['Last_Week_Influence'] = gws1.groupby(['id'])['influence'].shift().astype(float)
gws1['Last_Week_Minutes_Played'] = gws1.groupby(['id'])['minutes'].shift().astype(float)
gws1['Last_Week_Threat'] = gws1.groupby(['id'])['threat'].shift().astype(float)
gws1['Last_Week_Selected'] = gws1.groupby(['id'])['selected'].shift().astype(float)
gws1['Last_Week_Value'] = gws1.groupby(['id'])['value'].shift().astype(float)
gws1['Last_Week_Creativity'] = gws1.groupby(['id'])['creativity'].shift().astype(float)
gws1['Last_Week_FDR'] = gws1.groupby(['id'])['fdr'].shift().astype(float).astype(float)
gws1['Last_Week_Clean_Sheets'] = gws1.groupby(['id'])['clean_sheets'].shift().astype(float)
gws1['Last_Week_Goals_Conceded'] = gws1.groupby(['id'])['goals_conceded'].shift().astype(float)
gws1['Last_Week_Own_Goals'] = gws1.groupby(['id'])['own_goals'].shift().astype(float)
gws1['Last_Week_Penalties_Missed'] = gws1.groupby(['id'])['penalties_missed'].shift().astype(float)
gws1['Last_Week_Penalties_Saved'] = gws1.groupby(['id'])['penalties_saved'].shift().astype(float)
gws1['Last_Week_Red_Cards'] = gws1.groupby(['id'])['red_cards'].shift().astype(float)
gws1['Last_Week_Saves'] = gws1.groupby(['id'])['saves'].shift().astype(float)
gws1['Last_Week_Yellow_Cards'] = gws1.groupby(['id'])['yellow_cards'].shift().astype(float)
gws1.head()


# **Creating binary attributes for was_home**
# 

# In[68]:


#creating binary attributes for was_home

home=pd.get_dummies(gws1['home'])
gws1=pd.concat([gws1,home],axis=1)
gws1.head()


# **Calculating the correlation of the different variables with the dependent variable. This step was important to see what type of variables affect the dependent variable the most. Based on these results we were encouraged to feature engineer variables on those attributes that affect dependent variable the most**
# 

# In[69]:


#calculating the correaltion of the different variables with the dependent variable
c=gws1.corr().abs()
cor_target = c['total_points']
pd.set_option('display.max_rows', None)
cor_target.sort_values(ascending=False)


# **This is where the magic happens, we have feature engineered variables based on a number of factors like per 90 min, home, and away statistics. This is done to make the algorithm understand different patterns in the data. We will also utilize these variables to create visualizations.**

# In[70]:


#feature engineered a number of variables to better define the dependent variable
window=3

hpm =gws1.groupby('id',as_index=False)['Home Points'].rolling(window=window).sum().shift(1).astype(float)
gws1["Home Points Form"] = hpm.reset_index(level=0, drop=True)

apm =gws1.groupby('id',as_index=False)['Away Points'].rolling(window=window).sum().shift(1).astype(float)
gws1["Away Points Form"] = apm.reset_index(level=0, drop=True)

ppm =(gws1.groupby('id',as_index=False)['total_points'].expanding().sum()/gws1.groupby('id',as_index=False)['minutes'].expanding().sum()*90).astype(float)
gws1["Points_90"] = ppm.reset_index(level=0, drop=True)

vm =(gws1.groupby('id',as_index=False)['value'].expanding().sum()/gws1.groupby('id',as_index=False)['total_points'].expanding().sum()*90).astype(float)
gws1["Value_Point"] = vm.reset_index(level=0, drop=True)

apm =gws1.groupby('id',as_index=False)['assists'].expanding().sum()/gws1.groupby('id',as_index=False)['minutes'].expanding().sum().astype(float)
gws1["Assists per Min"] = apm.reset_index(level=0, drop=True)

gpm =(gws1.groupby('id',as_index=False)['goals_scored'].expanding().sum()/gws1.groupby('id',as_index=False)['minutes'].expanding().sum()*90).astype(float)
gws1["Goals_90"] = gpm.reset_index(level=0, drop=True)

ipm =(gws1.groupby('id',as_index=False)['influence'].expanding().sum()/gws1.groupby('id',as_index=False)['minutes'].expanding().sum()*90).astype(float)
gws1["Influence_90"] = ipm.reset_index(level=0, drop=True)

bpm =(gws1.groupby('id',as_index=False)['bonus'].expanding().sum()/gws1.groupby('id',as_index=False)['minutes'].expanding().sum()*90).astype(float)
gws1["Bonus_90"] = ipm.reset_index(level=0, drop=True)

ih =gws1.groupby('id',as_index=False)['Home Influence'].rolling(window=window).sum().shift(1).astype(float)
gws1["Home Influence Form"] = ih.reset_index(level=0, drop=True)

ah =gws1.groupby('id',as_index=False)['Away Influence'].rolling(window=window).sum().shift(1).astype(float)
gws1["Away Influence Form"] = ah.reset_index(level=0, drop=True)

hg =gws1.groupby('id',as_index=False)['Home Goals'].rolling(window=window).sum().shift(1).astype(float)
gws1["Home Goals Form"] = hg.reset_index(level=0, drop=True)

ag =gws1.groupby('id',as_index=False)['Away Goals'].rolling(window=window).sum().shift(1).astype(float)
gws1["Away Goals Form"] = hg.reset_index(level=0, drop=True)


gs =gws1.groupby('id',as_index=False)['goals_scored'].rolling(window=window).sum().shift(1).astype(float)
gws1["Goals Form"] = gs.reset_index(level=0, drop=True)

ag =gws1.groupby('id',as_index=False)['clean_sheets'].expanding().sum()/gws1.groupby('id',as_index=False)['minutes'].expanding().sum().astype(float)
gws1["Clean Sheets per Min"] = ag.reset_index(level=0, drop=True)

cs =gws1.groupby('id',as_index=False)['total_points'].expanding().sum()/gws1.groupby('id',as_index=False)['minutes'].expanding().sum().astype(float)
gws1["Points Per Min"] = cs.reset_index(level=0, drop=True)

gc =gws1.groupby('id',as_index=False)['goals_conceded'].expanding().sum()/gws1.groupby('id',as_index=False)['minutes'].expanding().sum().astype(float)
gws1["Goals Conceded Per Min"] = gc.reset_index(level=0, drop=True)

s =gws1.groupby('id',as_index=False)['saves'].expanding().sum()/gws1.groupby('id',as_index=False)['minutes'].expanding().sum().astype(float)
gws1["Save per min"] = s.reset_index(level=0, drop=True)

sh =gws1.groupby('id',as_index=False)['Home Saves'].rolling(window=window).sum().shift(1).astype(float)
gws1["Home Saves Form"] = sh.reset_index(level=0, drop=True)

sa =gws1.groupby('id',as_index=False)['Away Saves'].rolling(window=window).sum().shift(1).astype(float)
gws1["Away Saves Form"] = sa.reset_index(level=0, drop=True)

sf =gws1.groupby('id',as_index=False)['saves'].rolling(window=window).sum().shift(1).astype(float)
gws1["Saves Form"] = sf.reset_index(level=0, drop=True)

mp =gws1.groupby('id',as_index=False)['minutes'].rolling(window=window).sum().shift(1).astype(float)
gws1["Minutes Form"] = mp.reset_index(level=0, drop=True)

ch =gws1.groupby('id',as_index=False)['Home Creativity'].rolling(window=window).sum().shift(1).astype(float)
gws1["Home Creativity Form"] = ch.reset_index(level=0, drop=True)

ca =gws1.groupby('id',as_index=False)['Away Creativity'].rolling(window=window).sum().shift(1).astype(float)
gws1["Away Creativity Form"] = ca.reset_index(level=0, drop=True)

csh =gws1.groupby('id',as_index=False)['Home Clean Sheets'].rolling(window=window).sum().shift(1).astype(float)
gws1[" Home Clean Sheets Form"] = csh.reset_index(level=0, drop=True)

csa =gws1.groupby('id',as_index=False)['Away Clean Sheets'].rolling(window=window).sum().shift(1).astype(float)
gws1["Away Clean Sheets Form"] = csa.reset_index(level=0, drop=True)

cs =gws1.groupby('id',as_index=False)['clean_sheets'].rolling(window=window).sum().shift(1).astype(float)
gws1["Clean Sheets Form"] = cs.reset_index(level=0, drop=True)

bph =gws1.groupby('id',as_index=False)['Home Bonus'].rolling(window=window).sum().shift(1).astype(float)
gws1['Home Bonus Form'] = bph.reset_index(level=0, drop=True)

bp =gws1.groupby('id',as_index=False)['bonus'].rolling(window=window).sum().shift(1).astype(float)
gws1['Bonus Form'] = bp.reset_index(level=0, drop=True)

bpa =gws1.groupby('id',as_index=False)['Away Bonus'].rolling(window=window).sum().shift(1).astype(float)
gws1["Away Bonus Form"] = bpa.reset_index(level=0, drop=True)

icth =gws1.groupby('id',as_index=False)['Home ICT'].rolling(window=window).sum().shift(1).astype(float)
gws1['Home ICT Form'] = icth.reset_index(level=0, drop=True)

icta =gws1.groupby('id',as_index=False)['Away ICT'].rolling(window=window).sum().shift(1).astype(float)
gws1["Away ICT Form"] = icta.reset_index(level=0, drop=True)


ict =gws1.groupby('id',as_index=False)['ict_index'].rolling(window=window).sum().shift(1).astype(float)
gws1["ICT Form"] = ict.reset_index(level=0, drop=True)

ah =gws1.groupby('id',as_index=False)['Home Assists'].rolling(window=window).sum().shift(1).astype(float)
gws1['Home Assists Form'] = ah.reset_index(level=0, drop=True)

aa =gws1.groupby('id',as_index=False)['Away Assists'].rolling(window=window).sum().shift(1).astype(float)
gws1["Away Assists Form"] = aa.reset_index(level=0, drop=True)

ast =gws1.groupby('id',as_index=False)['assists'].rolling(window=window).sum().shift(1).astype(float)
gws1["Assists Form"] = ast.reset_index(level=0, drop=True)

bpss =gws1.groupby('id',as_index=False)['minutes'].expanding().sum()/gws1.groupby('id',as_index=False)['bps'].expanding().sum().astype(float)
gws1["Mins Per BPS"] = bpss.reset_index(level=0, drop=True)

gca =gws1.groupby('id',as_index=False)['Away Goals Conceded'].rolling(window=5).sum().shift(1).astype(float)
gws1["Away Goals Concecded Form"] = gca.reset_index(level=0, drop=True)

gch =gws1.groupby('id',as_index=False)['Home Goals Conceded'].rolling(window=5).sum().shift(1).astype(float)
gws1["Goals Conceded Home"] = gch.reset_index(level=0, drop=True)

gc =gws1.groupby('id',as_index=False)['goals_conceded'].rolling(window=5).sum().shift(1).astype(float)
gws1["Goals Conceded Form"] = gc.reset_index(level=0, drop=True)

gcm =gws1.groupby('id',as_index=False)['minutes'].expanding().sum()/gws1.groupby('id',as_index=False)['goals_conceded'].expanding().sum().astype(float)
gws1["Mins per Goals Conceded"] = gcm.reset_index(level=0, drop=True)

tp =gws1.groupby('id',as_index=False)['total_points'].rolling(window=window).sum().shift(1).astype(float)
gws1["Points Form"] = tp.reset_index(level=0, drop=True)

pm =gws1.groupby('id',as_index=False)['penalties_missed'].rolling(window=window).sum().shift(1).astype(float)
gws1["Penalties Missed Form"] = pm.reset_index(level=0, drop=True)

yc =gws1.groupby('id',as_index=False)['yellow_cards'].rolling(window=window).sum().shift(1).astype(float)
gws1["Yellow Cards Form"] = yc.reset_index(level=0, drop=True)

rc =gws1.groupby('id',as_index=False)['red_cards'].rolling(window=window).sum().shift(1).astype(float)
gws1["Red Cards Form"] = rc.reset_index(level=0, drop=True)


# **A function to capture the goal-scoring streak of a player. Basically, the counter will be set to '1' whenever a player scores a goal and keep on adding +1 to it as long as the player keeps scoring goals in consecutive matches. If the player fails to score a goal in a match then the counter will be set to 0 again.**
# 
# 
# 
# 

# In[71]:


# a function to capture the goal scoring streak of a player
def f(df):
    df['Goalstreak'] = (df['goals_scored'] > 0).cumsum()
    df['cumsum'] = np.nan
    df.loc[df['goals_scored'] == 0, 'cumsum'] = df['Goalstreak']
    df['cumsum'] = df['cumsum'].fillna(method='ffill')
    df['cumsum'] = df['cumsum'].fillna(0)
    df['Goal Streak'] = df['Goalstreak'] - df['cumsum']
    df.drop(['Goalstreak', 'cumsum'], axis=1, inplace=True)
    df['Goal Streak']=df['Goal Streak'].shift(1)
    return df


# In[72]:


gws1=gws1.groupby('id').apply(f)


# **Following the same concept explained above, we will capture the clean sheet streak of a player**
# 
# 

# In[73]:


def f2(df):
    df['csstreak'] = (df['clean_sheets'] > 0).cumsum()
    df['cumsum'] = np.nan
    df.loc[df['clean_sheets'] == 0, 'cumsum'] = df['csstreak']
    df['cumsum'] = df['cumsum'].fillna(method='ffill')
    df['cumsum'] = df['cumsum'].fillna(0)
    df['Clean Sheet Streak'] = df['csstreak'] - df['cumsum']
    df.drop(['csstreak', 'cumsum'], axis=1, inplace=True)
    df['Clean Sheet Streak']=df['Clean Sheet Streak'].shift(1)
    return df


# In[74]:


gws1=gws1.groupby('id').apply(f2)


# **A function to capture the penalty Saving streak of a player. These functions are important to analyze the patterns**
# 
# 

# In[75]:


def f3(df):
    df['psstreak'] = (df['penalties_saved'] > 0).cumsum()
    df['cumsum'] = np.nan
    df.loc[df['penalties_saved'] == 0, 'cumsum'] = df['psstreak']
    df['cumsum'] = df['cumsum'].fillna(method='ffill')
    df['cumsum'] = df['cumsum'].fillna(0)
    df['Penalty Save Streak'] = df['psstreak'] - df['cumsum']
    df.drop(['psstreak', 'cumsum'], axis=1, inplace=True)
    df['Penalty Save Streak']=df['Penalty Save Streak'].shift(1)
    return df


# In[76]:


gws1=gws1.groupby('id').apply(f3)


# **Dropping all the NAn, inf values created because of feature engineering. This is an important step because the algorithm is going to throw an error in there are any Nan's or if's present in the dataset**
# 

# In[77]:


#dropping all the NAn, inf values created because of feature engineering
gws1=gws1.replace([np.inf, -np.inf], np.nan)

gws1=gws1.dropna()
gws1.head()


# **We check the variance of each variable as it would affect the result.**

# In[78]:


#variance of each variable
gws1.var()


# In[79]:


from sklearn.model_selection import train_test_split,KFold,cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import mean_absolute_error,mean_squared_error, explained_variance_score,r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler,Binarizer,scale


# In[80]:


gws1.columns


# **After completing all the preprocessing steps we start with machine learning. We divide the dataset into X(Independent Varaibales) and y(Dependent Variable). We have to be very careful with the independent variables we choose because we can only train the model on those variables that will be available to us before the match is played**

# In[81]:


#importing various other libraries and defining X and y set for testing and training
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from collections import Counter
from sklearn.model_selection import RepeatedKFold
# from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_selection import RFE

X=gws1.drop(['name','points','total_points','ict_index','bps','goals_scored','assists','bonus','influence','minutes','threat','value','selected','creativity','own_goals','penalties_missed','penalties_saved','red_cards','yellow_cards','saves','selected','yellow_cards','clean_sheets','goals_conceded','home','id','fdr','points'
,'Away Points','Home Points','Away Influence','Home Influence','Away Creativity','Home Creativity','Home Goals','Away Goals','Home Saves','Away Saves','Away ICT','Home ICT','Away Bonus','Home Bonus','Away Clean Sheets','Home Clean Sheets',
'Away Penalties Saved','Home Penalties Saved','Away Goals Conceded','Home Goals Conceded','Away Assists','Home Assists'],axis=1)
y=gws1['points']


# **We have performed a feature selection to see which features are important in determining the independent variable. Note that in this part we have excluded all direct variables that are not available before the match is played.**

# In[82]:




names = list(X.columns.values)
 
#use linear regression as the model
lr = LinearRegression()
#rank all features, i.e continue the elimination until the last one
rfe = RFE(lr, n_features_to_select=1)
rfe.fit(X,y)
 
print ("Features sorted by their rank:")
sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names))


# **We again divide the dataset into X and y and plot a correlation heatmap to check multicollinearity. There should be no variable with a correlation of greater than 0.8 to rule out multicollinearity.**

# In[83]:


# from imblearn.oversampling import SMOTE
X1=gws1[['Clean Sheets per Min', 'Assists per Min', 'Save per min','Goals Conceded Per Min', 'Goals_90', 'Last_Week_Clean_Sheets', 'Penalty Save Streak', 'Last_Week_Red_Cards','Last_Week_Goals_Conceded','Last_Week_Goals','Last_Week_Penalties_Missed','Last_Week_Assists','Clean Sheets Form','Last_Week_Yellow_Cards','Last_Week_Value','Goals Conceded Form','Red Cards Form', 'Goals Form','Penalties Missed Form','Points Form','Last_Week_Own_Goals','Away Clean Sheets Form','Last_Week_FDR','Last_Week_Minutes_Played','Away Assists Form','Assists Form','Away Bonus Form','Yellow Cards Form','Mins Per BPS','Bonus_90']]
y1=y.astype(int)
plt.figure(figsize=(40,20))
sns.heatmap(X1.corr(),annot=True)


# **We have divided the datastet into training and testing set using a 80/20 split.** 

# In[84]:


X_train, X_test, y_train, y_test = train_test_split( X1, y1, test_size=0.20, random_state=150)


# **We cannot compare apples to oranges in the same way we cannot compare goals scored to goals saved. In order for everything to be comparable, we have scaled the values using a min-max scaler**

# In[85]:


scaler= MinMaxScaler(feature_range=(0, 1), copy=True)
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)



# **Fitting the algorithm with tuned hyperparameters**

# In[86]:



ovr=RandomForestClassifier(max_depth= 11, min_samples_leaf= 3,n_estimators=300, class_weight='balanced',max_features='sqrt',bootstrap=True)
ovr.fit(X_train, y_train)



# **Once the model is trained, we predict output for the testing set. Going a step further we can also see how sure the algorithm was before it made a decision by plotting probabilities**

# In[87]:


from sklearn.metrics import roc_curve
predict=ovr.predict(X_test)
predictp=ovr.predict_proba(X_test)


# **Storing the output as well as the probabilities for each prediction in a data frame**

# In[88]:


df1=pd.DataFrame({'Actual':y_test,'Predicted':predict})
a=df1[1]=predictp[:,0].round(decimals=2)
b=df1[2]=predictp[:,1].round(decimals=2)


df1.head()


# **Performance Metrics of the Model**

# In[89]:


from sklearn.metrics import classification_report, confusion_matrix,auc, roc_auc_score
from sklearn.metrics import precision_score

print(classification_report(y_test,predict))
print("Balanced Accuracy : ",balanced_accuracy_score(y_test,predict))
print("Confusion Matrix : \n", confusion_matrix(y_test,predict))
print("Model Score : ",ovr.score(X_test,y_test))


# **We achieved a balanced accuracy of 64%. There is a lot of work that still needs to be done improve the metrics for the minority class**

# 
# **Validation of unseen data. We try to predict scores for the upcoming week. We will be performing the same set of pre-processing on this dataset to make it consistent with the data on which the data was trained**

# In[90]:


conn = sqlite3.connect("/Users/anmolsabharwal/Desktop/FPL1.db")
cur = conn.cursor()
x= pd.read_sql_query("SELECT * FROM test", conn)
x.tail()
x = x.replace(r'^\s*$', np.NaN, regex=True)
x.columns


# In[91]:


x=x.astype({'assists':float})
x=x.astype({'bonus':float})
x=x.astype({'bps':float})
x=x.astype({'clean_sheets':float})
x=x.astype({'creativity':float})
x=x.astype({'element':float})
x=x.astype({'fixture':float})
x=x.astype({'goals_conceded':float})
x=x.astype({'goals_scored':float})
x=x.astype({'ict_index':float})
x=x.astype({'influence':float})
x=x.astype({'minutes':float})
x=x.astype({'opponent_team':float})
x=x.astype({'own_goals':float})
x=x.astype({'penalties_missed':float})
x=x.astype({'penalties_saved':float})
x=x.astype({'red_cards':float})
x=x.astype({'round':float})
x=x.astype({'selected':float})
x=x.astype({'team_a_score':float})
x=x.astype({'team_h_score':float})
x=x.astype({'threat':float})
x=x.astype({'total_points':float})
x=x.astype({'transfers_balance':float})
x=x.astype({'value':float})
x=x.astype({'yellow_cards':float})





# In[92]:


x = x.assign(id=(x['name']).astype('category').cat.codes)


# In[93]:


x.rename(columns={'opponent_team':'fdr'},inplace=True)


# In[94]:


x['fdr']=x['fdr'].apply(fd).astype(float)


# In[95]:


x['kickoff_time']=pd.to_datetime(x['kickoff_time'])


# In[96]:


x['kickoff_time'] = x.kickoff_time.dt.hour


# In[97]:


x['kickoff_time']=x['kickoff_time'].apply(time).astype(float)
x.tail()


# In[98]:


x['home']=x['was_home'].apply(was_home)
x.head()


# In[99]:


x['Away Points'] = np.where(x['home'] == 'Home', x['total_points'] * 0,x['total_points'])
x['Home Points'] = np.where(x['home'] == 'Away', x['total_points'] * 0,x['total_points'])

x['Away Influence'] = np.where(x['home'] == 'Home', x['influence'] * 0,x['influence'])
x['Home Influence'] = np.where(x['home'] == 'Away', x['influence'] * 0,x['influence'])

x['Away Goals'] = np.where(x['home'] == 'Home', x['goals_scored'] * 0,x['goals_scored'])
x['Home Goals'] = np.where(x['home'] == 'Away', x['goals_scored'] * 0,x['goals_scored'])

x['Away Saves'] = np.where(x['home'] == 'Home', x['saves'] * 0,x['saves'])
x['Home Saves'] = np.where(x['home'] == 'Away', x['saves'] * 0,x['saves'])

x['Away Creativity'] = np.where(x['home'] == 'Home', x['creativity'] * 0,x['creativity'])
x['Home Creativity'] = np.where(x['home'] == 'Away', x['creativity'] * 0,x['creativity'])

x['Away ICT'] = np.where(x['home'] == 'Home', x['ict_index'] * 0,x['ict_index'])
x['Home ICT'] = np.where(x['home'] == 'Away', x['ict_index'] * 0,x['ict_index'])

x['Away Bonus'] = np.where(x['home'] == 'Home', x['bonus'] * 0,x['bonus'])
x['Home Bonus'] = np.where(x['home'] == 'Away', x['bonus'] * 0,x['bonus'])

x['Away Clean Sheets'] = np.where(x['home'] == 'Home', x['clean_sheets'] * 0,x['clean_sheets'])
x['Home Clean Sheets'] = np.where(x['home'] == 'Away', x['clean_sheets'] * 0,x['clean_sheets'])

x['Away Penalties Saved'] = np.where(x['home'] == 'Home', x['penalties_saved'] * 0,x['penalties_saved'])
x['Home Penalties Saved'] = np.where(x['home'] == 'Away', x['penalties_saved'] * 0,x['penalties_saved'])

x['Away Goals Conceded'] = np.where(x['home'] == 'Home', x['goals_conceded'] * 0,x['goals_conceded'])
x['Home Goals Conceded'] = np.where(x['home'] == 'Away', x['goals_conceded'] * 0,x['goals_conceded'])

x['Away Assists'] = np.where(x['home'] == 'Home', x['assists'] * 0,x['assists'])
x['Home Assists'] = np.where(x['home'] == 'Away', x['assists'] * 0,x['assists'])

x.tail()


# In[100]:


x.drop(['element','fixture','round','transfers_balance','transfers_in','transfers_out','team_a_score','was_home',	'team_h_score','round'],axis=1, inplace=True)


# In[101]:


x['points']=x['total_points'].apply(points)
x.tail()


# In[102]:


home=pd.get_dummies(x['home'])
x=pd.concat([x,home],axis=1)
x.head()


# In[106]:


x['Last_Week_Points'] = x.groupby(['id'])['points'].shift().astype(float)
x['Last_Week_Goals'] = x.groupby(['id'])['goals_scored'].shift().astype(float)
x['Last_Week_ICT'] = x.groupby(['id'])['ict_index'].shift().astype(float)
x['Last_2_Week_ICT'] = x.groupby(['id'])['ict_index'].shift(2).astype(float)
x['Last_Week_BPS'] = x.groupby(['id'])['bps'].shift().astype(float)
x['Last_2_Week_BPS'] = x.groupby(['id'])['bps'].shift(2).astype(float)
x['Last_Week_Assists'] = x.groupby(['id'])['assists'].shift().astype(float)
x['Last_2_Week_Assists'] = x.groupby(['id'])['assists'].shift(2).astype(float)
x['Last_Week_Bonus'] = x.groupby(['id'])['bonus'].shift().astype(float)
x['Last_2_Week_Bonus'] = x.groupby(['id'])['bonus'].shift(2).astype(float)
x['Last_Week_Penalties_Saved'] = x.groupby(['id'])['penalties_saved'].shift().astype(float)
x['Last_2_Week_Penalties_Saved'] =x.groupby(['id'])['penalties_saved'].shift(2).astype(float)
x['Last_Week_Influence'] = x.groupby(['id'])['influence'].shift().astype(float)
x['Last_2_Week_Influence'] = x.groupby(['id'])['influence'].shift(2).astype(float)
x['Last_Week_Minutes_Played'] = x.groupby(['id'])['minutes'].shift().astype(float)
x['Last_2_Week_Minutes_Played'] = x.groupby(['id'])['minutes'].shift(2).astype(float)
x['Last_Week_Threat'] = x.groupby(['id'])['threat'].shift().astype(float)
x['Last_2_Week_Threat'] = x.groupby(['id'])['threat'].shift(2).astype(float)
x['Last_Week_Selected'] = x.groupby(['id'])['selected'].shift().astype(float)
x['Last_2_Week_Selected'] = x.groupby(['id'])['selected'].shift(2).astype(float)
x['Last_Week_Value'] = x.groupby(['id'])['value'].shift().astype(float)
x['Last_2_Week_Value'] = x.groupby(['id'])['value'].shift(2).astype(float)
x['Last_Week_Creativity'] = x.groupby(['id'])['creativity'].shift().astype(float)
x['Last_2_Week_Creativity'] = x.groupby(['id'])['creativity'].shift(2).astype(float)
x['Last_Week_FDR'] = x.groupby(['id'])['fdr'].shift().astype(float).astype(float)
x['Last_2_Week_FDR'] = x.groupby(['id'])['fdr'].shift(2).astype(float).astype(float)
x['Last_3_Week_FDR'] = x.groupby(['id'])['fdr'].shift(3).astype(float).astype(float)
x['Last_Week_Clean_Sheets'] = x.groupby(['id'])['clean_sheets'].shift().astype(float)
x['Last_2_Week_Clean_Sheets'] = x.groupby(['id'])['clean_sheets'].shift(2).astype(float)
x['Last_Week_Goals_Conceded'] = x.groupby(['id'])['goals_conceded'].shift().astype(float)
x['Last_2_Week_Goals_Conceded'] = x.groupby(['id'])['goals_conceded'].shift(2).astype(float)
x['Last_Week_Own_Goals'] = x.groupby(['id'])['own_goals'].shift().astype(float)
x['Last_2_Week_Own_Goals'] = x.groupby(['id'])['own_goals'].shift(2).astype(float)
x['Last_Week_Penalties_Saved'] = x.groupby(['id'])['penalties_saved'].shift().astype(float)
x['Last_Week_Penalties_Missed'] = x.groupby(['id'])['penalties_missed'].shift().astype(float)
x['Last_2_Week_Penalties_Missed'] = x.groupby(['id'])['penalties_missed'].shift(2).astype(float)
x['Last_Week_Red_Cards'] = x.groupby(['id'])['red_cards'].shift().astype(float)
x['Last_2_Week_Red_Cards'] = x.groupby(['id'])['red_cards'].shift(2).astype(float)
x['Last_Week_Saves'] = x.groupby(['id'])['saves'].shift().astype(float)
x['Last_2_Week_Saves'] = x.groupby(['id'])['saves'].shift(2).astype(float)
x['Last_Week_Yellow_Cards'] = x.groupby(['id'])['yellow_cards'].shift().astype(float)
x['Last_2_Week_Yellow_Cards'] = x.groupby(['id'])['yellow_cards'].shift(2).astype(float)
window=3
hpm =x.groupby('id',as_index=False)['Home Points'].rolling(window=window).sum().shift(1).astype(float)
x["Home Points Form"] = hpm.reset_index(level=0, drop=True)
apm =x.groupby('id',as_index=False)['Away Points'].rolling(window=window).sum().shift(1).astype(float)
x["Away Points Form"] = apm.reset_index(level=0, drop=True)
ppm =(x.groupby('id',as_index=False)['total_points'].expanding().sum()/x.groupby('id',as_index=False)['minutes'].expanding().sum()*90).astype(float)
x["Points_90"] = ppm.reset_index(level=0, drop=True)
vm =(x.groupby('id',as_index=False)['value'].expanding().sum()/x.groupby('id',as_index=False)['total_points'].expanding().sum()*90).astype(float)
x["Value_Point"] = vm.reset_index(level=0, drop=True)

apm =x.groupby('id',as_index=False)['assists'].expanding().sum()/x.groupby('id',as_index=False)['minutes'].expanding().sum().astype(float)
x["Assists per Min"] = apm.reset_index(level=0, drop=True)

gpm =(x.groupby('id',as_index=False)['goals_scored'].expanding().sum()/x.groupby('id',as_index=False)['minutes'].expanding().sum()*90).astype(float)
x["Goals_90"] = gpm.reset_index(level=0, drop=True)

ipm =(x.groupby('id',as_index=False)['influence'].expanding().sum()/x.groupby('id',as_index=False)['minutes'].expanding().sum()*90).astype(float)
x["Influence_90"] = ipm.reset_index(level=0, drop=True)

bpm =(x.groupby('id',as_index=False)['bonus'].expanding().sum()/x.groupby('id',as_index=False)['minutes'].expanding().sum()*90).astype(float)
x["Bonus_90"] = ipm.reset_index(level=0, drop=True)

ih =x.groupby(['id','name'],as_index=False)['Home Influence'].rolling(window=window).sum().shift(1).astype(float)
x["Home Influence Form"] = ih.reset_index(level=0, drop=True)

ah =x.groupby('id',as_index=False)['Away Influence'].rolling(window=window).sum().shift(1).astype(float)
x["Away Influence Form"] = ah.reset_index(level=0, drop=True)

hg =x.groupby('id',as_index=False)['Home Goals'].rolling(window=window).sum().shift(1).astype(float)
x["Home Goals Form"] = hg.reset_index(level=0, drop=True)

ag =x.groupby('id',as_index=False)['Away Goals'].rolling(window=window).sum().shift(1).astype(float)
x["Away Goals Form"] = hg.reset_index(level=0, drop=True)


gs =x.groupby('id',as_index=False)['goals_scored'].rolling(window=window).sum().shift(1).astype(float)
x["Goals Form"] = gs.reset_index(level=0, drop=True)

ag =x.groupby('id',as_index=False)['clean_sheets'].expanding().sum()/x.groupby('id',as_index=False)['minutes'].expanding().sum().astype(float)
x["Clean Sheets per Min"] = ag.reset_index(level=0, drop=True)

cs =x.groupby('id',as_index=False)['total_points'].expanding().sum()/x.groupby('id',as_index=False)['minutes'].expanding().sum().astype(float)
x["Points Per Min"] = cs.reset_index(level=0, drop=True)

gc =x.groupby('id',as_index=False)['goals_conceded'].expanding().sum()/x.groupby('id',as_index=False)['minutes'].expanding().sum().astype(float)
x["Goals Conceded Per Min"] = gc.reset_index(level=0, drop=True)

s =x.groupby('id',as_index=False)['saves'].expanding().sum()/x.groupby('id',as_index=False)['minutes'].expanding().sum().astype(float)
x["Save per min"] = s.reset_index(level=0, drop=True)

sf =x.groupby('id',as_index=False)['saves'].rolling(window=window).sum().shift(1).astype(float)
x["Saves Form"] = sf.reset_index(level=0, drop=True)

mp =x.groupby('id',as_index=False)['minutes'].rolling(window=window).sum().shift(1).astype(float)
x["Minutes Form"] = mp.reset_index(level=0, drop=True)

ch =x.groupby('id',as_index=False)['Home Creativity'].rolling(window=window).sum().shift(1).astype(float)
x["Home Creativity Form"] = ch.reset_index(level=0, drop=True)

ca =x.groupby('id',as_index=False)['Away Creativity'].rolling(window=window).sum().shift(1).astype(float)
x["Away Creativity Form"] = ca.reset_index(level=0, drop=True)

csh =x.groupby('id',as_index=False)['Home Clean Sheets'].rolling(window=window).sum().shift(1).astype(float)
x[" Home Clean Sheets Form"] = csh.reset_index(level=0, drop=True)

csa =x.groupby('id',as_index=False)['Away Clean Sheets'].rolling(window=window).sum().shift(1).astype(float)
x["Away Clean Sheets Form"] = csa.reset_index(level=0, drop=True)

cs =x.groupby('id',as_index=False)['clean_sheets'].rolling(window=window).sum().shift(1).astype(float)
x["Clean Sheets Form"] = cs.reset_index(level=0, drop=True)

bph =x.groupby('id',as_index=False)['Home Bonus'].rolling(window=window).sum().shift(1).astype(float)
x['Home Bonus Form'] = bph.reset_index(level=0, drop=True)

bp =x.groupby('id',as_index=False)['bonus'].rolling(window=window).sum().shift(1).astype(float)
x['Bonus Form'] = bp.reset_index(level=0, drop=True)

bpa =x.groupby('id',as_index=False)['Away Bonus'].rolling(window=window).sum().shift(1).astype(float)
x["Away Bonus Form"] = bpa.reset_index(level=0, drop=True)

icth =x.groupby('id',as_index=False)['Home ICT'].rolling(window=window).sum().shift(1).astype(float)
x['Home ICT Form'] = icth.reset_index(level=0, drop=True)

icta =x.groupby('id',as_index=False)['Away ICT'].rolling(window=window).sum().shift(1).astype(float)
x["Away ICT Form"] = icta.reset_index(level=0, drop=True)


ict =x.groupby('id',as_index=False)['ict_index'].rolling(window=window).sum().shift(1).astype(float)
x["ICT Form"] = ict.reset_index(level=0, drop=True)

ah =x.groupby('id',as_index=False)['Home Assists'].rolling(window=window).sum().shift(1).astype(float)
x['Home Assists Form'] = ah.reset_index(level=0, drop=True)

aa =x.groupby('id',as_index=False)['Away Assists'].rolling(window=window).sum().shift(1).astype(float)
x["Away Assists Form"] = aa.reset_index(level=0, drop=True)

ast =x.groupby('id',as_index=False)['assists'].rolling(window=window).sum().shift(1).astype(float)
x["Assists Form"] = ast.reset_index(level=0, drop=True)


bpss =x.groupby('id',as_index=False)['minutes'].expanding().sum()/x.groupby('id',as_index=False)['bps'].expanding().sum().astype(float)
x["Mins Per BPS"] = bpss.reset_index(level=0, drop=True)

gca =x.groupby('id',as_index=False)['Away Goals Conceded'].rolling(window=5).sum().shift(1).astype(float)
x["Away Goals Concecded Form"] = gca.reset_index(level=0, drop=True)

gch =x.groupby('id',as_index=False)['Home Goals Conceded'].rolling(window=5).sum().shift(1).astype(float)
x["Goals Conceded Home"] = gch.reset_index(level=0, drop=True)

gc =x.groupby('id',as_index=False)['goals_conceded'].rolling(window=5).sum().shift(1).astype(float)
x["Goals Conceded Form"] = gc.reset_index(level=0, drop=True)

gcm =x.groupby('id',as_index=False)['minutes'].expanding().sum()/x.groupby('id',as_index=False)['goals_conceded'].expanding().sum().astype(float)
x["Mins per Goals Conceded"] = gcm.reset_index(level=0, drop=True)

tp =x.groupby('id',as_index=False)['total_points'].rolling(window=window).sum().shift(1).astype(float)
x["Points Form"] = tp.reset_index(level=0, drop=True)

pm =x.groupby('id',as_index=False)['penalties_missed'].rolling(window=window).sum().shift(1).astype(float)
x["Penalties Missed Form"] = pm.reset_index(level=0, drop=True)

yc =x.groupby('id',as_index=False)['yellow_cards'].rolling(window=window).sum().shift(1).astype(float)
x["Yellow Cards Form"] = yc.reset_index(level=0, drop=True)

rc =x.groupby('id',as_index=False)['red_cards'].rolling(window=window).sum().shift(1).astype(float)
x["Red Cards Form"] = rc.reset_index(level=0, drop=True)

bpss =x.groupby('id',as_index=False)['minutes'].expanding().sum()/x.groupby('id',as_index=False)['bps'].expanding().sum().astype(float)
x["Mins Per BPS"] = bpss.reset_index(level=0, drop=True)

gcm =x.groupby('id',as_index=False)['minutes'].expanding().sum()/x.groupby('id',as_index=False)['goals_conceded'].expanding().sum().astype(float)
x["Mins per Goals Conceded"] = gcm.reset_index(level=0, drop=True)



# In[104]:


dx=x.groupby('id').apply(f)
x=x.groupby('id').apply(f2)
x=x.groupby('id').apply(f3)
x.head()


# **We seperate gameweek 150 from the rest of the data because we will be predicitng scores for that gameweek.**

# In[112]:


x1=x[x['gw']=='150']
x1
x_test=x1[['Clean Sheets per Min', 'Assists per Min', 'Save per min','Goals Conceded Per Min', 'Goals_90', 'Last_Week_Clean_Sheets', 'Penalty Save Streak', 'Last_Week_Red_Cards','Last_Week_Goals_Conceded','Last_Week_Goals','Last_Week_Penalties_Missed','Last_Week_Assists','Clean Sheets Form','Last_Week_Yellow_Cards','Last_Week_Value','Goals Conceded Form','Red Cards Form', 'Goals Form','Penalties Missed Form','Points Form','Last_Week_Own_Goals','Away Clean Sheets Form','Last_Week_FDR','Last_Week_Minutes_Played','Away Assists Form','Assists Form','Away Bonus Form','Yellow Cards Form','Mins Per BPS','Bonus_90']]
x_test=x_test.replace([np.inf, -np.inf], np.nan)
x_test=x_test.dropna()


# In[108]:


x_testf=x1[['name','Clean Sheets per Min', 'Assists per Min', 'Save per min','Goals Conceded Per Min', 'Goals_90', 'Last_Week_Clean_Sheets', 'Penalty Save Streak', 'Last_Week_Red_Cards','Last_Week_Goals_Conceded','Last_Week_Goals','Last_Week_Penalties_Missed','Last_Week_Assists','Clean Sheets Form','Last_Week_Yellow_Cards','Last_Week_Value','Goals Conceded Form','Red Cards Form', 'Goals Form','Penalties Missed Form','Points Form','Last_Week_Own_Goals','Away Clean Sheets Form','Last_Week_FDR','Last_Week_Minutes_Played','Away Assists Form','Assists Form','Away Bonus Form','Yellow Cards Form','Mins Per BPS','Bonus_90']]
x_testf=x_testf.replace([np.inf, -np.inf], np.nan)
x_testf=x_testf.dropna()
x_test.head()


# In[113]:


x_test1=scaler.transform(x_test)
predict1=ovr.predict(x_test1)


# In[110]:


predict1=pd.DataFrame(predict1)
predict1.head()
df1 = pd.concat([x_testf.reset_index(drop='True'),predict1.reset_index(drop='True')],axis=1)
df1.rename(columns={0:'Prediciton'},inplace=True)
df1.head()


# In[111]:


df1.to_sql('Predictions', conn, if_exists='replace', index=False)


# In[ ]:




