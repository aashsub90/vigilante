
# coding: utf-8

# # Data Cleaning

# This notebook is used to clean, reduce, transform and discretize the supporting data sets.

# In[1]:


import pandas as pd
import numpy as np
import pickle
import os


# ## Methods

# In[2]:


def loadData(file):
    '''
    Function: This function loads data from a csv file and return a pandas dataframe
    Parameters: Path to file to be loaded
    Returns: Pandas dataframe
    '''
    df = pd.read_csv(file)
    return df


# In[3]:


def storePickle(df,file):
    '''
    Function: This function pickles a pandas dataframe
    Parameters: Pandas dataframe and path of pickle file
    Returns: N/A
    '''
    df.to_pickle(file, compression='gzip', protocol=2)
    print("Pickling to file - %s complete." %file)


# In[4]:


def readPickle(file):
    '''
    Function: This function pickles a pandas dataframe
    Parameters: Pandas dataframe and path of pickle file
    Returns: N/A
    '''
    df = df.read_pickle(file, compression='gzip')
    print("Successfully read pickle file - %s to dataframe." %file)
    return df


# In[5]:


def preprocessData(df_list):
    for df_key in df_list.keys():
        if(df_key == 'facilities'):
            df = df_list[df_key][['common_name','address','longitude','latitude']]
            df = df.dropna()
            df['common_name'] = df['common_name'].apply(lambda x: x.lower().strip())
            df['address'] = df['address'].apply(lambda x: x.lower().strip())
            df.columns = ['name', 'address', "longitude", "latitude"]
            storePickle(df,'store/facilities.pkl')
        elif(df_key == 'private_open_spaces'):
            df = df_list[df_key][['NAME', 'the_geom', 'LOCATION']]
            df = df.dropna()
            df['Latitude'] = df['the_geom'].apply(lambda x: float(x[1:-1].split(",")[0].strip()))
            df['Longitude'] = df['the_geom'].apply(lambda x: float(x[1:-1].split(",")[1].strip()))
            df = df.drop(['the_geom'], axis=1)
            df['NAME'] = df['NAME'].apply(lambda x: x.lower().strip())
            df['LOCATION'] = df['LOCATION'].apply(lambda x: x.lower().strip())
            df.columns = ['address', 'name', "latitude", "longitude"]
            storePickle(df,'store/private_spaces.pkl')
        elif(df_key == 'colleges'):
            df = df_list[df_key][['Institution', 'Address', 'Location']]
            df = df.dropna()
            df['Latitude'] = df['Location'].apply(lambda x: float(x[1:-1].split(",")[0].strip()))
            df['Longitude'] = df['Location'].apply(lambda x: float(x[1:-1].split(",")[1].strip()))
            df['Institution'] = df['Institution'].apply(lambda x: x.lower().strip())
            df['Address'] = df['Address'].apply(lambda x: x.lower().strip())
            df = df.drop(['Location'], axis=1)
            df.columns = ['name', 'address', 'latitude', 'longitude']
            storePickle(df,'store/colleges.pkl')
        elif(df_key == 'public_open_spaces'):
            df = df_list[df_key][['ParkName', 'Location 1']]
            df = df.dropna()
            df['ParkName'] = df['ParkName'].apply(lambda x: x.lower().strip())
            df['Address'] = df["Location 1"].apply(lambda x: x.split("\n")[0].lower().strip())
            df['Latitude'] = df["Location 1"].apply(lambda x: float(x.split("\n")[2][1:-1].split(",")[0].strip()))
            df['Longitude'] = df["Location 1"].apply(lambda x: float(x.split("\n")[2][1:-1].split(",")[1].strip()))
            df = df.drop(['Location 1'], axis=1)
            df.columns = ['name', 'address', 'latitude', 'longitude']
            storePickle(df,'store/public_open_spaces.pkl')
        elif(df_key == 'commuter_stops'):
            df = df_list[df_key][['LOCATION', 'LATITUDE', 'LONGITUDE', 'PARKINGTYP']]
            df = df.dropna()
            df['PARKINGTYP'] = df['PARKINGTYP'].apply(lambda x: x.lower().strip()+" parking")
            df['LOCATION'] = df['LOCATION'].apply(lambda x: ','.join(x.lower().split(',')[0:2]))
            df.columns = ['address','latitude', 'longitude','name']
            storePickle(df,'store/commuter_stops.pkl')   
        elif(df_key == 'public_park'):
            df = df_list[df_key][['ParkName', 'Zipcode', 'Location 1']]
            df = df.dropna()
            df['ParkName'] = df['ParkName'].apply(lambda x: x.lower().strip())
            df['Address'] = df["Location 1"].apply(lambda x: x.split("\n")[0].lower().strip())
            df['Latitude'] = df["Location 1"].apply(lambda x: float(x.split("\n")[2][1:-1].split(",")[0].strip()))
            df['Longitude'] = df["Location 1"].apply(lambda x: float(x.split("\n")[2][1:-1].split(",")[1].strip()))
            df = df.drop(['Location 1','Zipcode'], axis=1)
            df.columns = ['name','address','latitude', 'longitude']
            storePickle(df,'store/public_park.pkl')  
        elif(df_key == 'landmarks'):
            df = df_list[df_key][['Name', 'the_geom']]
            df = df.dropna()
            df['Name'] = df['Name'].apply(lambda x: x.lower().strip())
            df['Latitude'] = df['the_geom'].apply(lambda x: float(x.strip('MULTIPOLYGON ')[3:-3].split(",")[0].split(" ")[1]))
            df['Longitude'] = df['the_geom'].apply(lambda x: float(x.strip('MULTIPOLYGON ')[3:-3].split(",")[1].split(" ")[1]))
            df = df.drop(['the_geom'],axis=1)
            df.columns = ['name','latitude', 'longitude']
            storePickle(df,'store/landmarks.pkl')
        elif(df_key == 'schools'):
            df = df_list[df_key][['Campus Name', 'Campus Address', 'Location 1']]
            df = df.dropna()
            df['Campus Name'] = df['Campus Name'].apply(lambda x:x.lower().strip())
            df['Campus Address'] = df['Campus Address'].apply(lambda x:x.split(',')[0].lower().strip())
            df['Latitude'] = df['Location 1'].apply(lambda x: float(x.split('\n')[1][1:-1].split(",")[0].strip()))
            df['Longitude'] = df['Location 1'].apply(lambda x: float(x.split('\n')[1][1:-1].split(",")[1].strip()))
            df = df.drop(['Location 1'], axis=1)
            df.columns = ['name', 'address', 'latitude', 'longitude']
            storePickle(df,'store/schools.pkl')
        else:
            pass


# In[6]:


if __name__ == "__main__":
    file_list = ['data/city_facilities_data.csv', 'data/privately_owned_public_open_spaces_data.csv', 'data/colleges_map_data.csv', 					'data/public_park_and_open_space_data.csv', 'data/commutershuttles_stops_data.csv', 'data/public_park_data.csv', 'data/landmarks_data.csv', 'data/schools_map_data.csv']
    print("\nNumber of files being loaded: %i" %(len(file_list)))
    df_dict = {}
    if not os.path.exists('store'):
        os.makedirs('store')
    
    df_dict['facilities'] = loadData(file_list[0])
    df_dict['private_open_spaces'] = loadData(file_list[1])
    df_dict['colleges'] = loadData(file_list[2])
    df_dict['public_open_spaces'] = loadData(file_list[3])
    df_dict['commuter_stops'] = loadData(file_list[4])
    df_dict['public_park'] = loadData(file_list[5])
    df_dict['landmarks'] = loadData(file_list[6])
    df_dict['schools'] = loadData(file_list[7])
    preprocessData(df_dict)

