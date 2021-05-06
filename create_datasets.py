import json
import os
import math
from datetime import datetime
import numpy as np
import pandas as pd
from random import sample 
#=============================================================================#
#------------------------------VARIABLES TO FILL IN---------------------------#
corner_time_margin = 500
sample_period = 250
save_dir = 'Signality/Model Data_6/'
non_corner_samples_num = 18500
#=============================================================================#
#-----------------------------------------------------------------------------#



def utc_to_epoch(date, format): 
  '''Convert datetime format to epoch in milliseconds.'''
  epoch = datetime.strptime(date, format).timestamp()
  return int(round(epoch * 1000.0))



def create_corner_dict(ground_truths_path):
  '''Go through ground truth file and record all corner events in a dict.'''
  corner_events = {}
  corner_counter = 0
  with open(ground_truths_path) as f:
    data = json.load(f)
   
    for game_id in data['game_id']:
      corner_events[game_id] = {}
      corner_events[game_id]['1'] = []
      corner_events[game_id]['2'] = []

      phases = ['1','2']

      for phase in phases:
        for event in data['game_id'][game_id][phase]:
          epoch_time = utc_to_epoch(event['utc_time'], '%Y-%m-%d %H:%M:%S.%f%z')
          corner_events[game_id][phase].append((epoch_time,event['team']))
          corner_counter += 1
      
  print('Total ground truth corners:', corner_counter)
  return corner_events


def load_game_data(game_id):
  '''Load info_live and track data from files.'''
  with open('Signality/2020/Tracking Data/'+game_id+'/'+game_id+'.1-info_live.json') as f:
    game_info = json.load(f)

  with open('Signality/2020/Tracking Data/'+game_id+'/'+game_id+'.1-tracks.json') as f:
    phase1_tracks = json.load(f)

  with open('Signality/2020/Tracking Data/'+game_id+'/'+game_id+'.2-tracks.json') as f:
    phase2_tracks = json.load(f)

  return game_info, phase1_tracks, phase2_tracks


def sample_data(event):
  '''Samples information in track file.

  Metrics sampled:
  mu_x, mu_y -- mean position of all players
  sigma_x, sigma_y -- standard devitation of player position
  mu_v -- speed of all players

  Returns a single numpy array
  '''
  home_team = event['home_team']
  away_team = event['away_team']

  home_array = np.array([d['position'] for d in home_team], dtype=np.float)
  away_array = np.array([d['position'] for d in away_team], dtype=np.float)

  # If the list contains only NaN values then ignore this sample.
  if np.all(np.isnan(home_array)) or np.all(np.isnan(away_array)): 
    return [np.nan]

  home_speed = np.array([d['speed'] for d in home_team], dtype=np.float)
  away_speed = np.array([d['speed'] for d in away_team], dtype=np.float)

  if np.all(np.isnan(home_speed)) or np.all(np.isnan(away_speed)): 
    return [np.nan]

  full_array = np.concatenate((home_array, away_array), axis=0)
  full_speed = np.concatenate((home_speed, away_speed), axis=0)

  # Ignore any NaN values in the calculation of the means.
  mean_pos = np.nanmean(full_array, axis=0) 
  pos_stds = np.std(full_array,axis=0)
  mean_speed = np.nanmean(full_speed)
  

  training_sample = np.concatenate((mean_pos, pos_stds, mean_speed), axis=None)
  
  return training_sample



# Main Function: Get corner data from given file and collect training & test
# data for our model.
detected_corners = {}
corner_events = create_corner_dict('corner-detection-challenge.json')

print(corner_events)

corners_processed = 0
corner_data = []
non_corner_data = []

for game_id in corner_events:

  game_info, phase1_tracks, phase2_tracks = load_game_data(game_id)

  start_time = utc_to_epoch(game_info['time_start'], '%Y-%m-%dT%H:%M:%S.%fZ') + 7200000
  
#   # It would be more proper to use this for regularization of player positions
#   # since pitch size can vary, but the difference in negligible.
#   pitch_size = game_info['calibration']['pitch_size']
#   pitch_x = pitch_size[0] / 2.0
#   pitch_y = pitch_size[1] / 2.0

  phases = ['1', '2']
  tracks = [phase1_tracks, phase2_tracks]

  for phase, track in zip(phases, tracks):
    
    corner_timestamps = corner_events[game_id][phase]
    num_corners = len(corner_timestamps)
    next_corner_idx = 0
    next_corner_time = corner_timestamps[next_corner_idx][0]
    previous_corner_time = start_time
    previous_non_corner_time = start_time

    for event in track:

      utc_time = event['utc_time']
      # Find nearest frame to archived corner event.
      if (abs(utc_time - next_corner_time) <= 200):
        training_sample = sample_data(event)
        if np.isnan(training_sample).any():
          continue
        corner_data.append(sample_data(event))
        corners_processed += 1
      elif (utc_time - next_corner_time > 200):
        # Update next corner if we're 200ms past the last one.
        previous_corner_time = next_corner_time
        next_corner_idx += 1
        if next_corner_idx == num_corners:
          next_corner_time = start_time + 999999999
        else:          
          next_corner_time = corner_timestamps[next_corner_idx][0]
      # Sample some non-corner events. Must not be too close to a corner event. 
      elif (abs(utc_time - next_corner_time) > corner_time_margin) and \
           (abs(utc_time - previous_corner_time) > corner_time_margin) and \
           (abs(utc_time - previous_non_corner_time) > sample_period):
        training_sample = sample_data(event)
        if np.isnan(training_sample).any():
          continue
        non_corner_data.append(sample_data(event))
        previous_non_corner_time = utc_time
    
    print(next_corner_idx)
#     print(len(non_corner_data))
print(corners_processed)
print(len(non_corner_data))

# Sample randomly from the non-corner data.
sampled_non_corners = sample(non_corner_data, non_corner_samples_num)

# Save as different files.
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
np.save(save_dir+'corner_data.npy', corner_data)
np.save(save_dir+'non_corner_data.npy', sampled_non_corners)
  