#This file contains steps to sue the Vigilante app on Slack

1. Create a python virtualenv

	virtualenv <env_name>

2. Source the new environment

	source <path_to_env>/bin/activate

3. Install dependencies using pip and the requirements file

	pip install -r requirements.txt

4. Create a Slack Bot and obtain the bot token [Ref - https://www.fullstackpython.com/blog/build-first-slack-bot-python.html]

5. Export the Bot token to a local environment variable:

	export SLACK_BOT_TOKEN='your bot user access token here'

6. Add the bot to a Slack workspace and invite the bot to a channel

7. Run init.sh script
	
	./init.sh

8. Start the bot service
	
	python app.py

9. Enter a command in the channel that contains the bot

	@<NameOfBot>Latitude, Longitude, District, Date as [yyyy-mm-dd] , Time as [hh:mm]
	
	eg. @vigilante 37.134,-12.443, northern, 2018-03-21,18:00

9. The bot will now respond with one of the 3 levels for safety based on the environment 
