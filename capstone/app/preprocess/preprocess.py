# pass in CLI args
# arg1 = directory of raw data files
# arg2 = chop size of audio file
# arg3 = number of chops per song

import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
import random
from os import listdir
from os.path import isfile, join



dir_path = '/content/drive/My Drive/CAPSTONE/test_data/'

def build_songs(dir_path):
  song_list = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
  return song_list

def load_song(song):
  # STEP 1: apply librosa sample load method
  samples, sampling_rate = librosa.load(dir_path + song,
                                        sr = 1600,
                                        mono = True,
                                        offset = 0.0,
                                        duration = None)
  return samples, sampling_rate, song

def extract_label(song):
  # STEP 1: split on hyphen
  labels = song.split('-')

  # STEP 2: select composer label
  composer = labels[-2]

  # STEP 3: select and clean era label
  era = labels[-1].split('.')[0]

  return composer, era

def chop_song(song, chop_size, num_of_chops):  
  # STEP 1: pull samples, sr and song name using load_song
  samples, sampling_rate, song = load_song(song)

  # STEP 2: extract labels
  composer, era = extract_label(song)

  # STEP 2: make dataframe from samples
  df_amps = pd.DataFrame(samples)
  
  # STEP 3: make time window by multiplying sampling rate by chop size (number of samples)
  time_window = int(sampling_rate * chop_size)

  # STEP 4: make empty dataframe to store chops as rows
  df_chops = pd.DataFrame()

  # STEP 5: for number of chops, extract random sample
  for _ in range(num_of_chops):
    # determine abs end of samples 
    end = df_amps.index[-1]
    # set last sample to be used in randomizer
    last = df_amps.index[end - time_window]
    # randomly select start point (integer between 0 and last)
    start_point = random.randrange(0, last, 1)
    # set end point as start point plus time window
    end_point = start_point + time_window
    
    # STEP 6: chop sample
    df_chop = df_amps[start_point:end_point]

    # STEP 7: convert to dataframe
    df_chop = df_chop.reset_index()
    df_chop = df_chop.drop(columns=['index'])
    df_chop = df_chop.transpose()
    df_chop['song'] = str(song)
    df_chop['sampling_rate'] = sampling_rate
    df_chop['composer'] = composer
    df_chop['label'] = era

    # STEP 8: add to all chops dataframe
    df_chops = df_chops.append(df_chop)

  return df_chops

def get_data(song_list, chop_size, num_of_chops):
  df_data = pd.DataFrame()
  num_of_songs = len(song_list)
  for song in song_list:
    df_chops = chop_song(song, chop_size, num_of_chops)
    df_data = df_data.append(df_chops)
    df_data = df_data.reset_index()
    df_data = df_data.drop(columns=['index'])

  # save data to csv
  csv_name = 'data/{}-songs-{}-chops-{}-seconds'.format(num_of_songs, num_of_chops, chop_size)
  
  # return dataframe locally
  return df_data, csv

df_data, csv = get_data(song_list=song_list, chop_size=5, num_of_chops=1)