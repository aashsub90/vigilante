
# coding: utf-8

# # Data Preprocessing

# This notebook is used to clean, reduce, transform, integrate and discretize the final data set.

# In[1]:


import pandas as pd
import numpy as np
from math import sin, cos, sqrt, atan2, radians
import pickle


# ## Methods

# In[2]:


def discretizeTime(time):
    '''
    function to map 24-hours format time to one of the six 4 hours intervals.
    '''
    if '00:00' <= time < '04:00':
        return 't1'
    elif '04:00' <= time < '08:00':
        return 't2'
    elif '08:00' <= time < '12:00':
        return 't3'
    elif '12:00' <= time < '16:00':
        return 't4'
    elif '16:00' <= time < '20:00':
        return 't5'
    elif '20:00' <= time < '24:00':
        return 't6'


# In[3]:


def categorizeMonth(month):
    '''
    function to map numeric months to their names.
    '''
    months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']
    return months[int(month)-1]


# In[4]:


def categorizeResolution(status):
    '''
    function to tag a resolution as No if the crime is not resolved, otherwise Yes.
    '''
    if status == 'NONE':
        return 'no'
    else:
        return 'yes'


# In[5]:


def binarize(value):
    '''
    function to binarize 'yes' to 1 and 'no' to 0.
    '''
    if value == 'yes':
        return 1
    elif value == 'no':
        return 0
    else:
        return value


# In[6]:


def calculateDistance(src, dst):
    '''
    function to calculate the distance between two locations on earth
    using src & dst tuples given in the format (latitude, longitude).
    '''
    # approximate radius of earth in km
    R = 6373.0

    # approximate 1 km to miles conversion
    to_miles = 0.621371

    lat1 = radians(abs(src[0]))
    lon1 = radians(abs(src[1]))
    lat2 = radians(abs(dst[0]))
    lon2 = radians(abs(dst[1]))

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c * to_miles


# In[7]:


def isNear(location, data, radius):
    '''
    function to determine if the given location (latitude, longitude)
    is near to any location in the given data (dataframe) based on the given radius.
    '''
    for index, row in data.iterrows():
        if calculateDistance(location, (row['latitude'], row['longitude'])) <= radius:
            return 'yes'
    return 'no'


# In[8]:


def labelCategory(category):
    '''
    function to label a data point as per crime category.
    '''
    low = ["trespass", "drunkenness", "runaway", "family offenses", "trea", "liquor laws", "warrants", "other offenses", "sex offenses, non forcible"]
    moderate = ["arson", "driving under the influence", "stolen property", "prostitution", "recovered vehicle", "suspicious occ", "pornography/obscene mat" , "disorderly conduct"]
    high = ["vehicle theft", "weapon laws", "vandalism", "assault", "robbery", "sex offenses, forcible", "missing person", "larceny/theft", "kidnapping", "fraud", "extortion", "burglary", "drug/narcotic" ]
    if category in low:
        return 'low'
    elif category in moderate:
        return 'moderate'
    elif category in high:
        return 'high'


# In[9]:


def integrate_data(crime_data, data_name):
    '''
    function to integrate support data sets to the crime data.
    '''
    data = pd.read_pickle('store/'+data_name+'.pkl', compression='gzip')
    column_data = crime_data.apply(lambda row: isNear((row['latitude'], row['longitude']), data, 1), axis=1)
    crime_data['near_'+data_name] = column_data
    return crime_data


# In[10]:


def integrate_dataframes(crime_data, data_name):
    '''
    function to integrate support data sets to the crime data.
    '''
    data = pd.read_pickle('store/crime_data_'+data_name+'.pkl', compression='gzip')
    crime_data['near_'+data_name] = data['near_'+data_name]
    return crime_data


# ## Process

# In[11]:


crime_data = pd.read_csv('data/crime_data.csv')


# Clean the data. Replace 'nan' values with 'N/A'. Drop columns that do not help with the goal. Remove rows that do not fall under the goal criteria.

# In[12]:


crime_data = crime_data.replace(np.nan, 'N/A')
crime_data = crime_data.drop(['IncidntNum', 'Descript', 'Location', 'PdId', 'Address'], axis=1)
crime_data = crime_data[crime_data.PdDistrict != 'N/A']
crime_data = crime_data[crime_data.Category != 'NON-CRIMINAL']
crime_data = crime_data[crime_data.Category != 'SECONDARY CODES']
crime_data = crime_data[crime_data.Category != 'GAMBLING']
crime_data = crime_data[crime_data.Category != 'FORGERY/COUNTERFEITING']
crime_data = crime_data[crime_data.Category != 'LOITERING']
crime_data = crime_data[crime_data.Category != 'SUICIDE']
crime_data = crime_data[crime_data.Category != 'BAD CHECKS']
crime_data = crime_data[crime_data.Category != 'EMBEZZLEMENT']
crime_data = crime_data[crime_data.Category != 'BRIBERY']


# Modify the column names and values to match the scenario and neccesity.

# In[13]:


crime_data = crime_data.rename(str.lower, axis='columns')
crime_data = crime_data.rename(index=str, columns={"dayofweek": "day", "pddistrict": "district", "x": "longitude", "y": "latitude"})
crime_data['category'] = crime_data['category'].apply(str.lower)
crime_data['day'] = crime_data['day'].apply(str.lower)
crime_data['district'] = crime_data['district'].apply(str.lower)


# Split the 'date' column into 'month' and 'year' for better classification.

# In[14]:


date = crime_data['date'].str.split('/')
month = date.apply(lambda x: x[0])
year = date.apply(lambda x: x[2])
crime_data['month'] = month
crime_data['month'] = crime_data['month'].apply(categorizeMonth)
crime_data['year'] = year
crime_data = crime_data.drop('date', axis=1)


# Discretize 'time' column to be represented using 6 interval classes.

# In[15]:


time_interval = crime_data['time'].apply(discretizeTime)
crime_data['time_interval'] = time_interval
crime_data = crime_data.drop('time', axis=1)


# Create a 'resolved' column to represent if a crime report was resolved or not.

# In[16]:


resolved = crime_data['resolution'].apply(categorizeResolution)
crime_data['resolved'] = resolved
crime_data = crime_data.drop('resolution', axis=1)


# Externally label the data points to reflect what is to be achieved.

# In[17]:


label = crime_data['category'].apply(labelCategory)
crime_data['label'] = label


# Integrate support datasets into crime data.

# In[18]:


support_data = ['facilities','private_spaces','colleges','public_open_spaces','commuter_stops','public_park','landmarks','schools']
for data in support_data:
    crime_data = integrate_data(crime_data, data)
# support_data = ['facilities','private_spaces','colleges','public_open_spaces','commuter_stops','public_park','landmarks','schools']
# for data in support_data:
#     crime_data = integrate_dataframes(crime_data, data)


# Binarize the 'yes' and 'no' values.

# In[19]:


crime_data = crime_data.applymap(binarize)


# Save the dataframe as a pickle to store directory.

# In[20]:


crime_data.to_pickle('store/crime_data.pkl', compression='gzip', protocol=2)
print("Pickling to file - crime_data.pkl complete.")

