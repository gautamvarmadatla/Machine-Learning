import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


dataset = pd.read_csv(r'D:\SHOWS\heart_failure_clinical_records_dataset.csv')
print(dataset)

# number of rows and columns
dataset.shape



# Identifying if any null data points and type of data in each column
dataset.info()

dataset.isnull().sum()

# ____________________________________________________________________________________________
# Selecting most important contributing features using extra tree classifier and removing the outliers
plt.rcParams['figure.figsize']=15,6
sns.set_style("darkgrid")

x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(x,y)
print(model.feature_importances_)
feat_importances= pd.Series(model.feature_importances_, index=x.columns)
feat_importances.nlargest(12).plot(kind='barh')
plt.show()


# We will select only 3 features : time, ejection_fraction, serum_creatinine
# Box plot to identify outliers
sns.boxplot(x = dataset.ejection_fraction, color ='red')
plt.show()

sns.boxplot(x=dataset.time, color ='blue')
plt.show()

sns.boxplot(x=dataset.serum_creatinine, color ='green')
plt.show()

# Removing outliers from boxplox i.e rejecting all value above 70
dataset = dataset[dataset['ejection_fraction'] < 70]

# _________________________________________________________________________________
# Numerical data distribution of features
# i) Age Distribution:
print("Average age of the data set is ", dataset['age'].mean())
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Histogram(
    x=dataset['age'],
    xbins=dict(start=25, end=100, size=1), marker_color='#60e8bd', opacity=0.5))

fig.update_layout(
    title_text='Age distribution', xaxis_title_text='Age', yaxis_title_text='Count', bargap=0.06, xaxis={'showgrid': False}, yaxis={'showgrid': False},
    template='plotly_dark')
fig.show()

# ii) Creatinine Phospokinase Distribution:

import plotly.graph_objects as go


fig = go.Figure()
fig.add_trace(go.Histogram(
    x = dataset['creatinine_phosphokinase'],
    xbins=dict(
        start=20,
        end=590,
        size=15
    ),
    marker_color='#FE6F5E',
    opacity=1
))

fig.update_layout(
    title_text='CREATININE PHOSPHOKINASE DISTRIBUTION',
    xaxis_title_text='CREATININE PHOSPHOKINASE',
    yaxis_title_text='COUNT',
    bargap=0.05,
    xaxis =  {'showgrid': False },
    yaxis = {'showgrid': False },
    template = 'plotly_dark'
)

fig.show()




# iii) Ejection Fraction Distribution

import plotly.graph_objects as go
fig.add_trace(go.Histogram(
    x = dataset['ejection_fraction'],
    xbins=dict(
        start=10,
        end=80,
        size=2
    ),
    marker_color='#A7F432',
    opacity=1
))

fig.update_layout(
    title_text='EJECTION FRACTION DISTRIBUTION',
    xaxis_title_text='EJECTION FRACTION',
    yaxis_title_text='COUNT',
    bargap=0.05,
    xaxis =  {'showgrid': False },
    yaxis = {'showgrid': False },
    template = 'plotly_dark'
)

fig.show()


# iv) Platelets distribution

import plotly.graph_objects as go

fig = go.Figure()


fig = go.Figure()
fig.add_trace(go.Histogram(
    x = dataset['platelets'],
    xbins=dict(
        start=25000,
        end=300000,
        size=5000
    ),
    marker_color='#50BFE6',
    opacity=1
))

fig.update_layout(
    title_text='PLATELETS DISTRIBUTION',
    xaxis_title_text='PLATELETS',
    yaxis_title_text='COUNT',
    bargap=0.05,
    xaxis =  {'showgrid': False },
    yaxis = {'showgrid': False },
    template = 'plotly_dark'
)

fig.show()

# v)Serum Creatinine Distribution

import plotly.graph_objects as go


fig = go.Figure()
fig.add_trace(go.Histogram(
    x = dataset['serum_creatinine'],
    xbins=dict(
        start=0.5,
        end=9.4,
        size=0.2
    ),
    marker_color='#E77200',
    opacity=1
))

fig.update_layout(
    title_text='SERUM CREATININE DISTRIBUTION',
    xaxis_title_text='SERUM CREATININE',
    yaxis_title_text='COUNT',
    bargap=0.05,
    xaxis =  {'showgrid': False },
    yaxis = {'showgrid': False },
    template = 'plotly_dark'
)

fig.show()


# vi) Serum sodium distribution

import plotly.graph_objects as go


fig = go.Figure()
fig.add_trace(go.Histogram(
    x = dataset['serum_sodium'],
    xbins=dict( # bins used for histogram
        start=113,
        end=148,
        size=1
    ),
    marker_color='#AAF0D1',
    opacity=1
))

fig.update_layout(
    title_text='SERUM SODIUM DISTRIBUTION',
    xaxis_title_text='SERUM SODIUM',
    yaxis_title_text='COUNT',
    bargap=0.05, # gap between bars of adjacent location coordinates
    xaxis =  {'showgrid': False },
    yaxis = {'showgrid': False },
    template = 'plotly_dark'
)
fig.show()

# So far we plotted distribution for those features whose values were numerical in nature , for non-numerical features ,
# i have used the pi plot

# vii) Gender distribution

import plotly.graph_objects as go
from plotly.subplots import make_subplots

d1 = dataset[(dataset["DEATH_EVENT"] == 0) & (dataset["sex"] == 1)]
d2 = dataset[(dataset["DEATH_EVENT"] == 1) & (dataset["sex"] == 1)]
d3 = dataset[(dataset["DEATH_EVENT"] == 0) & (dataset["sex"] == 0)]
d4 = dataset[(dataset["DEATH_EVENT"] == 1) & (dataset["sex"] == 0)]

Gender = ["Male", "Female"]
Gender_deaths = ['Male - Survived', 'Male - Died', "Female -  Survived", "Female - Died"]
V1 = [(len(d1) + len(d2)), (len(d3) + len(d4))]
V2 = [len(d1), len(d2), len(d3), len(d4)]


fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]])

fig.add_trace(go.Pie(labels=Gender, values=V1, name="Gender"), 1, 1)
fig.add_trace(go.Pie(labels=Gender_deaths, values=V2, name="Gender vs death event"), 1, 2)


fig.update_traces(hole=.4, hoverinfo="label+percent")

fig.update_layout(title_text="GENDER DISTRIBUTION IN THE DATASET  \ GENDER VS DEATH_EVENT",

    annotations=[dict(text='GENDER', x=0.19, y=0.5, font_size=10, showarrow=False),
                 dict(text='GENDER VS DEATH_EVENT', x=0.84, y=0.5, font_size=9, showarrow=False)],
    autosize=False, width=1200, height=500, paper_bgcolor="white")

fig.show()


# viii) Diabetes
import plotly.graph_objects as go
from plotly.subplots import make_subplots

d1 = dataset[(dataset["DEATH_EVENT"] ==0) & (dataset["diabetes"]==0)]
d2 = dataset[(dataset["DEATH_EVENT"]==0) & (dataset["diabetes"]==1)]
d3 = dataset[(dataset["DEATH_EVENT"]==1) & (dataset["diabetes"]==0)]
d4 = dataset[(dataset["DEATH_EVENT"]==1) & (dataset["diabetes"]==1)]

Gender = ["No Diabetes", "Diabetes"]
label2 = ['No Diabetes - Survived','Diabetes - Survived', "No Diabetes -  Died", "Diabetes  - Died"]
V1 = [(len(d1) + len(d3)), (len(d2) + len(d4))]
V2 = [len(d1), len(d2), len(d3), len(d4)]

# Create subplots: use 'domain' type for Pie subplot
fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
fig.add_trace(go.Pie(labels=Gender, values=V1, name="DIABETES"),
              1, 1)
fig.add_trace(go.Pie(labels=label2, values=V2, name="DIABETES VS DEATH_EVENT"),
              1, 2)

# Use `hole` to create a donut-like pie chart
fig.update_traces(hole=.4, hoverinfo="label+percent")

fig.update_layout(
    title_text="DIABETES DISTRIBUTION IN THE DATASET \
                  DIABETES VS DEATH_EVENT",
    # Add annotations in the center of the donut pies.
    annotations=[dict(text='DIABETES', x=0.20, y=0.5, font_size=10, showarrow=False),
                 dict(text='DIABETES VS DEATH_EVENT', x=0.84, y=0.5, font_size=8, showarrow=False)],
    autosize=False,width=1200, height=500, paper_bgcolor="white")
fig.show()


# ix) Smoking
import plotly.graph_objects as go
from plotly.subplots import make_subplots

d1 = dataset[(dataset["DEATH_EVENT"]==0) & (dataset["smoking"]==0)]
d2 = dataset[(dataset["DEATH_EVENT"]==1) & (dataset["smoking"]==0)]
d3 = dataset[(dataset["DEATH_EVENT"]==0) & (dataset["smoking"]==1)]
d4 = dataset[(dataset["DEATH_EVENT"]==1) & (dataset["smoking"]==1)]

Gender = ["No Smoking", "Smoking"]
label2 = ['No Smoking - Survived','No Smoking - Died', "Smoking - Survived", "Smoking - Died"]
V1 = [(len(d1) + len(d2)), (len(d3) + len(d4))]
V2 = [len(d1), len(d2), len(d3), len(d4)]

# Create subplots: use 'domain' type for Pie subplot
fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
fig.add_trace(go.Pie(labels=Gender, values=V1, name="SMOKING"),
              1, 1)
fig.add_trace(go.Pie(labels=label2, values=V2, name="SMOKING VS DEATH_EVENT"),
              1, 2)

# Use `hole` to create a donut-like pie chart
fig.update_traces(hole=.4, hoverinfo="label+percent")

fig.update_layout(
    title_text="SMOKING DISTRIBUTION IN THE DATASET \
                  SMOKING VS DEATH_EVENT",
   # Add annotations in the center of the donut pies.
    annotations=[dict(text='SMOKING', x=0.20, y=0.5, font_size=10, showarrow=False),
                 dict(text='SMOKING VS DEATH_EVENT', x=0.84, y=0.5, font_size=8, showarrow=False)],
    autosize=False,width=1200, height=500, paper_bgcolor="white")
fig.show()


# _______________________________________________________________________________________________________________
# Splitting the input data into test and train data
# I used sklearn library train_test_split in which we input the test size and split the data accordingly


x = dataset.iloc[:, [4, 7, 11]].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.6, random_state=0)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# _____________________________________________________________________________________________________________
# Applying logistic regression from sklearn

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

print(y_test)
print(y_pred)
# In the output window we can see both test and predicted outputs using which we can figure out at what values
# our machine made an error

# ___________________________________________________________________________________________________________
# To measure the parameters we create confusion matrix and the test and train data respectively.
mylist = []
from sklearn.metrics import confusion_matrix, accuracy_score
confusion_m = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
mylist.append(ac)
print(confusion_m)


regression = LogisticRegression(max_iter = 1000)
regression.fit(x_train, y_train )
print("Train Accuracy:",regression.score(x_train, y_train))
print("Test Accuracy:",regression.score(x_test, y_test))

#______________________________________________________________________________________________________________








