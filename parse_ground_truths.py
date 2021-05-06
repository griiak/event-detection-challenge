import json
import os
import math
from datetime import datetime
from scipy.spatial import distance

#=============================================================================#
#------------------------------VARIABLES TO FILL IN---------------------------#
MIN_DIST = 25
#=============================================================================#
#-----------------------------------------------------------------------------#



def utc_to_epoch(date, format):
  epoch = datetime.strptime(date, format).timestamp()
  return int(round(epoch * 1000.0))



def create_corner_dict(ground_truths_path):
  corner_events = {}

  with open(ground_truths_path) as f:
    data = json.load(f)
    
    for game_id in data['game_id']:
      corner_events[game_id] = {}
      corner_events[game_id]['1'] = []
      corner_events[game_id]['2'] = []

      for event in data['game_id'][game_id]['1']:
        epoch_time = utc_to_epoch(event['utc_time'], '%Y-%m-%d %H:%M:%S.%f%z')
        corner_events[game_id]['1'].append((epoch_time,event['team']))

  return corner_events


def is_ball_in_corner(ball_pos, pitch_size, margin):
  ball_x = abs(ball_pos[0])
  ball_y = abs(ball_pos[1])
  pitch_x = pitch_size[0]
  pitch_y = pitch_size[1]

  is_corner = False

  # print(ball_x)
  # print(ball_y)
  # print(pitch_x)
  # print(pitch_y)

  if (abs(ball_x - pitch_x) <= margin) and (abs(ball_y - pitch_y) <= margin):
    is_corner = True

  return is_corner



def is_corner(event, pitch_size, margin):
  ball_pos = event['ball']['position']
  
  if ball_pos == None:
    return False
  elif is_ball_in_corner(ball_pos, pitch_size, margin):
    return True

  return False

def load_game_data(game_id):
  with open('Signality/2020/Tracking Data/'+game_id+'/'+game_id+'.1-info_live.json') as f:
    game_info = json.load(f)

  with open('Signality/2020/Tracking Data/'+game_id+'/'+game_id+'.1-tracks.json') as f:
    phase1_tracks = json.load(f)

  with open('Signality/2020/Tracking Data/'+game_id+'/'+game_id+'.2-tracks.json') as f:
    phase2_tracks = json.load(f)

  return game_info, phase1_tracks, phase2_tracks


def ball_near_corner(ball_pos, pitch_corners):
  
  near_corner = False
  min_dist = 9999
  min_corner = None
  
  # print(ball_x)
  # print(ball_y)
  # print(pitch_x)
  # print(pitch_y)
  for idx, corner in enumerate(pitch_corners):
    dist = distance.euclidean(ball_pos, corner)
    if dist < min_dist:
      min_dist = dist
      min_corner = idx


  if (min_dist < MIN_DIST):
    near_corner = True

  return near_corner, min_dist, min_corner

def nearest_player_to_corner(event, corner):
  nearest_player = None
  nearest_team = None
  min_dist = 9999
  
  away_players = event['away_team']
  home_players = event['home_team']

  teams = [away_players, home_players]
  team_strings = ['away_team', 'home_team']

  for team, team_string in zip(teams, team_strings):
    for player in team:
      #print(player['position'])
      pos = player['position'] 
      dist = distance.euclidean(pos, corner)
      if dist < min_dist:
        min_dist = dist
        nearest_player = player['position']
        nearest_team = team_string

  
  # print(min_dist)
  # print(nearest_player)
  # print(nearest_team)

  return min_dist, nearest_team, nearest_player

detected_corners = {}
corner_events = create_corner_dict('corner-detection-challenge.json')

print(corner_events)

corner_counter = 0

for game_id in corner_events:
  detected_corners[game_id] = []
  # for corner in corner_events[game_id]:
  #   print(corner)

  game_info, phase1_tracks, phase2_tracks = load_game_data(game_id)

  print(game_id)
  print(game_info['time_start'])
  start_time = utc_to_epoch(game_info['time_start'], '%Y-%m-%dT%H:%M:%S.%fZ') + 7200000
  pitch_size = game_info['calibration']['pitch_size']

  print(start_time)
  print(pitch_size)

  pitch_x = pitch_size[0] / 2.0
  pitch_y = pitch_size[1] / 2.0

  pitch_corners = [[-pitch_x, -pitch_y], [pitch_x, -pitch_y], [-pitch_x, pitch_y], [pitch_x, pitch_y]]

  # input('...')
  ball_is_lost = False
  lost_ball_counter = 0
  possible_corner = False

  for event in phase1_tracks:

    ball_pos = event['ball']['position']

    if ball_pos == None:
      if not ball_is_lost:
        ball_is_lost = True
      lost_ball_counter += 1
    else:
      if lost_ball_counter > 0:
        ball_pos = ball_pos[:2]
        #print(ball_pos)
        lost_ball_counter = 0
        ball_is_lost = False
        near_corner, corner_dist, nearest_corner = ball_near_corner(ball_pos, pitch_corners)
        
        if near_corner:
          player_dist, player_team, player_pos = nearest_player_to_corner(event,pitch_corners[nearest_corner])      
          if player_dist < corner_dist:
            possible_corner = True
            previous_corner = nearest_corner
            previous_ball_pos = ball_pos
            previous_dist = corner_dist

      else:

        if possible_corner:
          possible_corner = False
          ball_pos = ball_pos[:2]
          new_dist = distance.euclidean(ball_pos, nearest_corner)
          if new_dist > previous_dist:
            print("Detected Corner")
            print("Ball position:", ball_pos)
            print("Corner distance:", corner_dist)
            print("Nearest corner:", nearest_corner)
            print("Timestamp: ", event['utc_time'])
            print("Player distance:", player_dist)
            print("Nearest team:", player_team)
            print("Player position:", player_pos)
            print("\n\n")
            detected_corners[game_id].append({event['utc_time']-1000})
            corner_counter += 1
   
    #print(event)


  game_info = None
  phase1_tracks = None
  phase2_tracks = None


  print('Corner counter:',corner_counter)
  print(detected_corners)
  exit(0)

# root_dir = 'Signality/2020/Tracking Data'

# for subdir, dirs, files in os.walk(root_dir):
#   for dir in dirs:
#     print(os.path.join(subdir,dir))

# with open('corner-detection-challenge.json') as f:
#   data = json.load(f)