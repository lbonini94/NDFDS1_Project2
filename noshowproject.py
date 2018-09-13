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


