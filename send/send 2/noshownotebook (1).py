
# coding: utf-8

# 
# # Project: Investigate show up appointments in Brazil.
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# ### Dataset characteristics
# This dataset collects information from 100k medical appointments in Brazil and is focused on the question of whether patients show up for their appointment. Several characteristics about the patient are included in each row.
# 
# ### Data Dictionary
# 
# - PatientId - Identification of a patient 
# - AppointmentID - Identification of each appointment
# - Gender = Male or Female. 
#     - Female is the greater proportion, woman takes way more care of they health in comparison to man. 
# - ScheduledDay = The day of the actual appointment, when they have to visit the doctor. 
# - AppointmentDay = The day someone called or registered the appointment, this is before appointment of course.
# - Age = How old is the patient. 
# - Neighbourhood = Where the appointment takes place. 
# - Scholarship = Ture of False. 
#     - Indicates whether or not the patient is enrolled in Brasilian welfare program Bolsa Família.
# - Hipertension = True or False 
# - Diabetes = True or False 
# - Alcoholism = True or False 
# - Handcap =  Number of disabilities a person has. (1, 2, 3, 4)
# - SMS_received = 1 or more messages sent to the patient. 
# - No-show = True or False.
#     - Be careful about the encoding of the last column: it says ‘No’ if the patient showed up to their appointment, and ‘Yes’ if they did not show up.
# 
# Source: https://www.kaggle.com/joniarroba/noshowappointments
# 

# <a id='wrangling'></a>
# ## Data Wrangling
# 
# ### General Properties

# In[339]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[340]:


df = pd.read_csv('noshow.csv')  # Reading file
df.head()


# In[341]:


df.info()


# - Some variables format can be changed for better analysis
# - No missing values

# In[342]:


df.shape #rows x columns


# In[343]:


df.nunique()  #How many unique values there are


# ### Data Cleaning

# In[344]:


#Drop PatientID and AppointmentID
df.drop(['PatientId', 'AppointmentID'], axis=1, inplace=True)


# - PatientID is less than AppointmentID which indicates that many patients scheduled more than one visit in the period
#     - but this will not be analyzed in this work

# In[345]:


#Rename Columns
df.rename(columns=lambda x: x.lower(), inplace=True)
df.rename(index=str, columns={"no-show": "no_show"}, inplace=True)

df.head()


# In[346]:


#Change Data Type

df['no_show'] = df.no_show.replace({'Yes': 1, 'No': 0})
#df['gender'] = df.gender.replace({'M': 1, 'F': 0})

df['scheduledday'] = pd.to_datetime(df.scheduledday)
df['appointmentday'] = pd.to_datetime(df.appointmentday)

df.info()


# In[347]:


df.hist(figsize=(8, 8));


# - I believe that the most important thing here is that there are more people who attend than those who do not.
# - The analysis will be made upon the people who do not attend.

# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# ### Which age group has the highest attendance?

# In[348]:


df.age[df.no_show == 0].hist(alpha=0.5, label='show up')  #age histogram of those who attend
df.age[df.no_show == 1].hist(alpha=0.5, label='no show')  #age histogram of those who do not
plt.legend();
plt.xlabel('Age');
plt.ylabel('People');
plt.title('Histogram by age');


# - Primary look

# In[349]:


df.describe().age


# In[350]:


df.loc[df['age'] < 0]   #locate ages less than zero


# Probably a mistake.
# - If it were a pattern of unborn babies there would be more. 
# - Therefore, this data will be excluded.

# In[351]:


df.drop(df.index[99832], inplace=True)  #delete row


# In[352]:


df.loc[df['age'] < 0]  #verify


# In[353]:


df.describe().age


# In[354]:


#Create a new column with age stages
bin_edges = [0, 9, 16, 25, 35, 50, 65, 75, 115]
bin_names = ['0 - 9', '10 - 16', '17 - 25', '26 - 35', '36 - 50', '51 - 65', '65 - 75', '76 - 115']

df['age_stages'] = pd.cut(df['age'], bin_edges, labels=bin_names)

df.head()


# - Creating age groups

# In[355]:


print(df.shape, '\n')
df.count()


# In[356]:


df.isnull().sum()  #verify null values


# In[357]:


df[df.age_stages.isnull()].head()   #show some null values


# - For some reason these data are not entering the groups.
#     - The zero will be replaced by a near and negative value.

# In[358]:


#Recreate column 
bin_edges = [-0.1, 9, 16, 25, 35, 50, 65, 75, 115]
bin_names = ['Child(0-9)', 
             'Adolescent(10-16)', 
             'Young(17 - 25)', 
             'Adult(26-35)', 
             'Mature(36-50)', 
             'Ageing(51-65)', 
             'Old(65-75)', 
             'Elderly(76-115)']

df['age_stages'] = pd.cut(df['age'], bin_edges, labels=bin_names)

df.head()


# In[359]:


df.isnull().sum()  #verify null values


# In[360]:


df.no_show.sum() # all no-show persons


# In[361]:


"""
sum of persons who do not attend in each age group divided by the total sum of persons who do not attend
"""
df_age = (df.groupby('age_stages').sum().no_show / df.no_show.sum())*100
df_age


# - Percentage only over people who do not attend.

# In[362]:


df_age.plot(kind='bar');
plt.ylabel('%');
plt.title('% Not attend by age group');


# ### Which neighborhood shows the highest attendance rate?

# In[363]:


"""
sum of people who do not attend per neighborhood divided by the total sum of people who do not attend.
"""
df_neigh = (df.groupby('neighbourhood').sum().no_show / df.no_show.sum())*100
df_neigh.describe()


# In[364]:


type(df_neigh)


# In[365]:


a = df_neigh.sum() - df_neigh.nlargest(5).sum() #TOP 5 neighbourhoods
a


# In[366]:


#Grouping the neighborhoods with the lowest percentage in an 'others' group.
df_neigh = df_neigh.nlargest(5)
df_neigh = df_neigh.append(pd.Series(a, index=['OTHERS']))
df_neigh


# - Show only the 5 neighborhoods with the highest rates of non-attendance.
# - The others are grouped as 'others'

# In[367]:


df_neigh.plot.pie(y='neighbourhood', figsize=(8, 8));
plt.pie(df_neigh, autopct='%1.0f%%', shadow=True);
plt.title('% Not attend by neighbourhood');


# ### Which gender has the highest attendance?
# 
# - Since it has been said in advance that there is a greater proportion of females, it is attempted to readjust to a more coherent view.

# In[368]:


M = df[df['gender'] == 'M'].shape #how many men there are
M[0]


# In[369]:


F = df[df['gender'] == 'F'].shape #how many women there are
F[0]


# In[370]:


df_gen = df.groupby('gender').sum().no_show # how many did not attend by gender
df_gen


# In[371]:


(df_gen[0]/F[0])*100 #those who did not attend in proportion


# In[372]:


(df_gen[1]/M[0])*100 #those who did not attend in proportion


# In[373]:


df_gen = pd.Series([(df_gen[0]/F[0])*100, (df_gen[1]/M[0])*100], index=['Female', 'Male'])
df_gen


# In[374]:


df_gen.plot(kind='bar');
plt.ylabel('%');
plt.title('% Who did not attend in proportion to gender');


# - It is noted that, in due proportions, there is  is an insignificant difference between genders.

# ### Do people who receive SMS tend to attend more?
# 
# - The same proportion adjustment was made in this section

# In[375]:


a = df[df['sms_received'] == 1].shape #how many people have received sms
a[0]


# In[376]:


b = df[df['sms_received'] == 0].shape #how many people did not received sms
b[0]


# In[377]:


df_sms = df.groupby('sms_received').sum().no_show
df_sms


# In[378]:


df_sms = pd.Series([(df_sms[1]/a[0])*100, (df_sms[0]/b[0])*100], index=['Not Received', 'Received'])
df_sms


# In[379]:


df_sms.plot(kind='bar');
plt.ylabel('%');
plt.title('% Who did not attend in proportion to SMS received');


# - Here, there is a reduction in the rate of non-attendance by almost 11% among people who received SMS.

# ### People receiving Bolsa Família show up more?
# >The same proportion adjustment was made in this section

# In[380]:


a = df[df['scholarship'] == 1].shape #how many people have received scholarship
a[0]


# In[381]:


b = df[df['scholarship'] == 0].shape #how many people did not received scholarship
b[0]


# In[382]:


df_scholar = df.groupby('scholarship').sum().no_show
df_scholar


# In[383]:


df_scholar = pd.Series([(df_scholar[0]/b[0])*100, (df_scholar[1]/a[0])*100], index=['Not Received', 'Received'])
df_scholar


# In[384]:


df_scholar.plot(kind='bar');
plt.ylabel('%');
plt.title("% Who did not attend in proportion to 'Scholarship' received");


# ### The greater the number of disabilities, the greater the chances of attending?
# >To avoid repetition, here was done in a different way.

# In[385]:


df_han = df.groupby('handcap').sum().no_show #sum of persons with disabilities and dot not attend
df_han


# - There are many more people who do not have some kind of disability.

# In[386]:


for i in range(5):
    print(df[df['handcap'] == i].count().no_show) #sum of persons disabilities


# In[387]:


for i in range(5):
    df_han[i] = (df_han[i] / df[df['handcap'] == i].count().no_show)*100
df_han


# In[388]:


df_han.plot(kind='bar');
plt.ylabel('%');
plt.title("% Who did not attend in proportion to number of disabilities");
plt.text(-1.5,-6.5,"Total People");
plt.text(-0.2,-6.5,"108285");
plt.text(0.8,-6.5,"2042");
plt.text(1.9,-6.5,"183");
plt.text(2.95,-6.5,"13");
plt.text(4,-6.5,"3");
plt.xlabel('');


# ### What month were there more appearances?
# Ref: https://www.youtube.com/watch?v=yCgJGsg0Xa4

# In[389]:


#time between appointments
df.appointmentday.max() - df.appointmentday.min()


# In[390]:


#data collecting time
df.appointmentday.max() - df.scheduledday.min()


# >Number of days data was collected

# In[391]:


print('Schedule')
print(df.scheduledday.min())
print(df.scheduledday.max())
print('-------------------------')
print('Appointment')
print(df.appointmentday.min())
print(df.appointmentday.max())


# In[392]:


df.scheduledday.max() - df.scheduledday.min()


# - Here we have an incompatibility.
#     - Since the "appointment day" column has no time, there are probably appointments scheduled on the same day, but all data was recorded at the end of the day.
#     - So for a better understanding, let's add 23:59 hours to all data.

# In[393]:


#Adding 23:59 hours in each row
appointmentday = []
for value in df.appointmentday:
    value += pd.Timedelta('+23:59:00')
    appointmentday.append(value)
df['appointmentday'] = appointmentday
df.appointmentday.head()


# >https://pandas.pydata.org/pandas-docs/stable/timedeltas.html

# In[394]:


df.appointmentday.max() - df.scheduledday.min() #verify


# > Before was: Timedelta ('210 days 16:46:04')

# In[395]:


#Creating two columns with respective scheduled and appointment months
df['apmonth'] = df.appointmentday.dt.month
df['scmonth'] = df.scheduledday.dt.month
df.head()


# In[396]:


#number of scheduled no-show appointments per month
df_sc = df.groupby('scmonth').sum().no_show
#df_sc = pd.DataFrame(df_sc)
df_sc


# In[397]:


#number of scheduled appointments per month
sc = []
for i in range(1,7):
    sc.append(df[df['scmonth'] == i].count().no_show)
for i in range(11,13):
    sc.append(df[df['scmonth'] == i].count().no_show)
sc


# In[398]:


#divide to see percentage
df_sch = (df_sc.values / sc)*100
df_sch


# In[399]:


#transforming in data-frame
df_sch = pd.DataFrame(df_sch, index=['03', '04', '05', '06', '07', '08', '01', '02'], columns=['No-show'])
df_sch


# In[400]:


df_sch = df_sch.sort_index() #reorder by month

#change lables
labels = ['Nov/15', 'Dec/15', 'Jan/16', 'Feb/16', 'Mar/16', 'Apr/16', 'May/16', 'Jun/16']
df_sch.index = labels
df_sch


# In[401]:


df_sch.plot(kind='bar');

plt.title('% No-show appointments by month')
plt.ylabel('%');
plt.text(-2.5,-9.5,"Total Scheduled:");
plt.text(0,-9.5,"1");
plt.text(0.8,-9.5,"61");
plt.text(1.9,-9.5,"60");
plt.text(2.95,-9.5,"281");
plt.text(3.8,-9.5,"3614");
plt.text(4.7,-9.5,"25339");
plt.text(5.7,-9.5,"67421");
plt.text(6.8,-9.5,"13749");


# - Percentage of attendances in relation to the number of appointments scheduled in each month.
#     - In Nov / 15, there was an attendance in the single appointment scheduled that month, resulting in 0%

# ### What about hipertesion, diabetes and alcoholic people?

# In[402]:


df_des = (df.groupby(['hipertension', 'diabetes', 'alcoholism']).sum().no_show / 
        df.groupby(['hipertension', 'diabetes', 'alcoholism']).count().no_show)*100
df_des


# In[403]:


#total people with the disease
df.groupby(['hipertension', 'diabetes', 'alcoholism']).count().no_show


# In[404]:


df_des.plot(kind='bar');
plt.ylabel('%');
plt.title('% Sick people who do not attend by type of illness');


# <a id='conclusions'></a>
# ## Conclusions

# Taking an analysis considering the **proportion** of the data collected, we arrive at some interesting conclusions such as:
#    - People aged 36 to 65 tend not to attend more than others.
#    
#    - The neighborhoods JARDIM CAMBURI, MARIA ORTIZ, ITARARÉ, RESISTÊNCIA and CENTRO in general have the highest rates of non-attendance, without considering scheduling proportions.
#    
#    - Although women schedule much more consultations, gender is not a factor in the issue of attendance.
#    
#    - People who receive prior SMS tend to attend more.
#    
#    - People who receive Bolsa Família assistance tend **not** to attend.
#    
#    - People with more deficiencies tend **not** to attend.
#    
#    - Although the number of appointments scheduled increase over the months, with the peak in May, the percentage of people who do not appear is **decreasing**.
#    
#    - There are a large number of people with hypertension, diabetes and alcohol problems. Given the proportions, 17 to 20% of people who have none or all three types of problems tend **not** to attend.
#    
# #### Sugestions
#  - The main suggestion is to send more alerts by SMS and maybe other types of alerts can also be created.
#  - In addition:
#     - Alternative means of alerting people with disabilities may be considered.
#     - Any kind of fee could be charged.
#     - To supervise the reasons of the people who receive Bolsa Família are missing the consultations.
#     - Use some means in which the person can tell if he can attend, giving way to another.
#    
