# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 14:37:48 2019

@author: Fran
"""

"""============================================================================
===============================================================================
Script that gets the live match data from Signality's API and stores it in the
format and folders needed for doing live analyses during matches.

The data is stored in the folders "Tracking Data/" and 
"Twelve-Analysis-Tools/Tracking Data/"
===============================================================================
============================================================================"""

game_halves = ['.1','.2']

import json
import requests
import gzip
import shutil
import os


# Get list of relevant games
with open('corner-detection-challenge.json') as f:
  games_to_fetch = json.load(f)


# Credentials for Signality API
SIGNALITY_API = "https://api.signality.com"
USERNAME = 'interview' 
PASSWORD = 'fdf>+~By8h8D)5)' 


save_dir1 = 'Signality/2020/Tracking Data/'
if not os.path.exists(save_dir1):
        os.makedirs(save_dir1)

# Get access token
payload = json.dumps({"username": USERNAME, "password": PASSWORD})
headers = {"Content-Type": "application/json"}
response = requests.post(SIGNALITY_API + "/v3/users/login", data=payload, headers=headers)
response = response.json()
token = response["id"]
user_id = response["userId"]

# Get game id
header = {"Authorization": token, "Content-Type": "application/json"}
response = requests.get(SIGNALITY_API + f"/v3/users/{user_id}/games", headers=header)
available_games = response.json()





for game in available_games:
    if game['id'] in games_to_fetch['game_id']:
        game_id = game['id']

        print("Fetching game :", game_id)

        save_dir2 = game_id+'/'
        if not os.path.exists(save_dir1+save_dir2):
            os.makedirs(save_dir1+save_dir2)



        # Get phase id
        header = {"Authorization": token, "Content-Type": "application/json"}
        response = requests.get(SIGNALITY_API+"/v3/games/"+game_id+'/phases', headers=header)
        available_phases = response.json()
        
        for game_half in game_halves:
            if game_half=='.1':
                phase_id = available_phases[0]['id']
            elif game_half=='.2':
                phase_id = available_phases[1]['id']
            
            # Download files
            response = requests.get(SIGNALITY_API+"/v3/games/"+game_id+'?filter=%7B%22include%22%3A%22calibration%22%7D', headers=header)
            info_live = response.json()
            datafile_name = save_dir1+save_dir2+game_id+game_half+'-info_live.json'
            with open(datafile_name, "w") as write_file:
                json.dump(info_live,write_file)
        
        
            files_list = ['events','tracks','stats']
            
            for file in files_list:
                response = requests.get(SIGNALITY_API+"/v3/games/"+game_id+'/phases/'+phase_id+'/'+file, headers=header)
                filename = save_dir1+save_dir2+game_id+game_half+'-'+file+'.json'
                with open(filename, "wb") as f:
                    f.write(response.content)