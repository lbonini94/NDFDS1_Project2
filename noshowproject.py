"""
Investigate Data
Name: Lucas Amorim Bonini
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

#Definitions


####Load data####
df = pd.read_csv("noshow.csv")

####Acessing Data#### 

print(df.head())
input("\nPress Enter to continue... \n")

#print(df.shape)
#(110527, 14)

print(df.info())
input("\nPress Enter to continue... \n")

print(df.nunique())
input("\nPress Enter to continue... \n")

####Cleaning####

#Drop PatientID and AppointmentID
df.drop(['PatientId', 'AppointmentID'], axis=1, inplace=True)

#Rename Columns
df.rename(columns=lambda x: x.lower(), inplace=True)
df.rename(index=str, columns={"no-show": "no_show"}, inplace=True)

print('**Droped and Renamed Columns**')
input("Press Enter to continue...\n")

#Change Data Type

df['no_show'] = df.no_show.replace({'Yes': True, 'No': False})
#df['gender'] = df.gender.replace({'M': 1, 'F': 0})

df['scheduledday'] = pd.to_datetime(df.scheduledday)
df['appointmentday'] = pd.to_datetime(df.appointmentday)

print(df.nunique())
input("\nPress Enter to continue... \n")


print('**Data Type fixed**')
input("Press Enter to continue... \n\n")


#confirm
print(df.head(20))

input("\nPress Enter to continue... \n")

###Age####
print(df.describe().age)

bin_edges = [-1, 18, 35, 60, 90, 120]
bin_names = ['Unborn - 18', '18 - 35', '35 - 60', '60 - 90', '90 - 120']

df['age_stages'] = pd.cut(df['age'], bin_edges, labels=bin_names)

print(df.no_show.sum())

print(df.groupby('age_stages').sum().no_show)
