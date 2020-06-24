import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import librosa
from librosa import display
import scipy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from score import score_obs

# header
st.title("Classical Music Era Predictor")  # h1
st.header("A music information retrieval project that predicts composition era of classical music using audiofile feature extraction.")  # h2
st.subheader("Sara Soueidan")  # h3

# add song file
#st.header("Drop Your Song")
#song = st.file_uploader("Upload a Song", type=["mp3"])
song = 'data/sample.mp3'

# play song
# audio_bytes = song.read()
# st.audio(audio_bytes, format='audio/mp3')

# waveplot
y, sr = librosa.load(song)
librosa.display.waveplot(y  = y,
                         sr = sr)
st.pyplot()

# harmonic and percussive waveplot
y_harm, y_perc = librosa.effects.hpss(y)
librosa.display.waveplot(y_harm, sr=sr,x_axis=None, cmap='OrRd', alpha=0.25)
librosa.display.waveplot(y_perc, sr=sr,x_axis=None, cmap='PuRd', alpha=0.5)

st.pyplot()

# chromagram
S = np.abs(librosa.stft(y=y, n_fft=4096))**2 
chromagram = librosa.feature.chroma_stft(S=S, sr=sr)
librosa.display.specshow(chromagram, cmap='viridis')

st.pyplot()

# 