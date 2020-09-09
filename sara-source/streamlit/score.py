import streamlit as st
import numpy as np
from sklearn.linear_model import LogisticRegression

# make it so the model doesn't have to fit every time
# add cache decorator
# make sure you include a doc string

@st.cache
def score_obs(model, observation):
    """
    predicts the user's song genre
    Args:
        model (sklearn model object): already fit model
        observation (ndarray): chromagram
    Returns:
        prediction (str): song era
    """
    return model.predict(observation.reshape(1, -1))