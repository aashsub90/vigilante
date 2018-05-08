import os
import time
import re
from slackclient import SlackClient
import pickle
import sklearn
import datetime
import pandas as pd
import numpy as np
from math import sin, cos, sqrt, atan2, radians

# instantiate Slack client
slack_client = SlackClient(os.environ.get('SLACK_BOT_TOKEN'))
# starterbot's user ID in Slack: value is assigned after the bot starts up
bot_id = None

# constants
RTM_READ_DELAY = 1 # 1 second delay between reading from RTM
EXAMPLE_COMMAND = "do"
MENTION_REGEX = "^<@(|[WU].+?)>(.*)"

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

def categorizeMonth(month):
    '''
    function to map numeric months to their names.
    '''
    months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']
    return months[int(month)-1]

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

def isNear(location, data, radius):
    '''
    function to determine if the given location (latitude, longitude)
    is near to any location in the given data (dataframe) based on the given radius.
    '''
    for index, row in data.iterrows():
        if calculateDistance(location, (row['latitude'], row['longitude'])) <= radius:
            return 1
    return 0

def integrate_data(input_data, data_name):
    '''
    function to integrate support data sets to the crime data.
    '''
    data = pd.read_pickle('store/'+data_name+'.pkl', compression='gzip')
    column_data = input_data.apply(lambda row: isNear((row['latitude'], row['longitude']), data, 1), axis=1)
    input_data['near_'+data_name] = column_data
    return input_data

def processDate(date):

    date = [int(i) for i in date.split("-")]
    year = date[0]
    month = categorizeMonth(date[1])
    date = datetime.datetime(date[0],date[1],date[2])
    days=["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
    day_number=date.weekday()
    day = days[day_number]
    return month, day 

def loadLabelEncoder(column):
    '''
    function to load label encoder object for the given column.
    '''
    with open('store/'+column+'_label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    return le

def encode(input_data):
	cols = ['day', 'district', 'month', 'time_interval']
	for column in cols:
		le = loadLabelEncoder(column)
		input_data[column] = le.transform(input_data[column])
	return input_data

def loadModel():
	"""
		Loads final model
	"""
	with open('store/randomForest_model.pkl', 'rb') as f:
		model = pickle.load(f)
	return (model)

def parse_bot_commands(slack_events):
    """
        Parses a list of events coming from the Slack RTM API to find bot commands.
        If a bot command is found, this function returns a tuple of command and channel.
        If its not found, then this function returns None, None.
    """
    for event in slack_events:
        if event["type"] == "message" and not "subtype" in event:
            user_id, message = parse_direct_mention(event["text"])
            if user_id == bot_id:
                return message, event["channel"]
    return None, None

def parse_direct_mention(message_text):
    """
        Finds a direct mention (a mention that is at the beginning) in message text
        and returns the user ID which was mentioned. If there is no direct mention, returns None
    """
    matches = re.search(MENTION_REGEX, message_text)

    # the first group contains the username, the second group contains the remaining message
    return (matches.group(1), matches.group(2).strip()) if matches else (None, None)

def genDataPoint(input):

	# Get the individual features from user input
	features = [feature.strip() for feature in input.strip().split(",")]

	# Assign variables for each feature
	latitude = features[0]
	longitude = features[1]
	district = features[2]
	resolved = 0
	date = features[3]
	time_interval = discretizeTime(features[4])
	month, day = processDate(date)

	# Create a dataframe using features
	input_data = pd.DataFrame(data = np.array([day, district, longitude, latitude, month, time_interval, resolved]).reshape(1,-1),columns=['day', 'district', 'longitude', 'latitude', 'month', 'time_interval', 'resolved'])
    

    # Convert data types to float
	input_data['longitude'] = input_data['longitude'].astype(np.float64)
	input_data['latitude'] = input_data['latitude'].astype(np.float64)
	input_data['resolved'] = input_data['resolved'].astype(np.int64)
    # Integrate data from different datasets
	support_data = ['facilities','private_spaces','colleges','public_open_spaces','commuter_stops','public_park','landmarks','schools']
	for data in support_data:
		input_data = integrate_data(input_data, data)

    # Encode columns with LabelEncoder
	data_point = encode(input_data)
	return (data_point)

def predict(data, model):
	"""
		Obtain model prediction
	"""
	# Obtain prediction value
	prediction = model.predict(data)
	
	# Load label encoder
	le = loadLabelEncoder('label')

	# Obtain original label for the prediction
	if len(list(prediction)) > 0:
		prediction = le.inverse_transform(list(prediction))
	else:
		prediction = None

	if(prediction[0] == 'low'):
		response = "This is a SAFE zone."
	elif(prediction[0] == 'moderate'):
		response = "Stay vigilant..this is a MODERATE zone."
	elif(prediction[0] == 'high'):
		response = "BEWARE! This region has been found to be a crime hotspot."
	else:
		response = "Sorry..I'm can't predict right now."
	return (response)



def handle_command(command, channel, model):
    """
        Executes bot command if the command is known
    """
    response = None
    input_data = genDataPoint(command)

    try:
    	response = predict(input_data, model)
    except:
    	response = "Looks like you entered incorrect input. Please adhere to the format - Latitude, Longitude, District, Date [yyyy-mm-dd] , Time [hh:mm]"

    # Finds and executes the given command, filling in response
    # This is where you start to implement more commands!
    # Sends the response back to the channel
    slack_client.api_call(
        "chat.postMessage",
        channel=channel,
        text=response
    )

if __name__ == "__main__":
    if slack_client.rtm_connect(with_team_state=False):
        
        print("Vigilante connected and running!")

        # Read bot's user ID by calling Web API method `auth.test`
        bot_id = slack_client.api_call("auth.test")["user_id"]

        # Load model 
        model = loadModel()

        # Listen on channel
        while True:
            command, channel = parse_bot_commands(slack_client.rtm_read())
            if command:
                handle_command(command, channel, model)
            time.sleep(RTM_READ_DELAY)
    else:
        print("Connection failed. Exception traceback printed above.")
