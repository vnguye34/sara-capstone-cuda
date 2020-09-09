**Problem Statment**

Is it possible to differentiate era of composition for "classical" music pieces (pre-1900) that sound phonically similar to the human ear?

**Workflow / To Do List**

1. flatten directory
2. process all .m4a (or additional files) using Librosa library - retrieve two objects
  - Samples (array of amplitudes for sound at specific sequential point in time)
  - Sampling rate or sampling fz (how many amplitude values are retrieved per second)
3. Chop up the samples into 10-30s chunks (proportional to length of song total)
4. Add chops to DF + columns for sampling rate, track title (long variable text), and era
  - era identification: find / search for composer - each composer == specific era
  - manually add era to end of tracks (timely, but more accurate - can select specific pieces for eras in which composer straddles multiple eras)

**Move directly into  NN - amplitudes == tensors == vectors**

OR

**Feature engineering and use images - CNN**

1. waveplot / frequency diagram 
2. Fourier Transform - decomposes resultant amplitudes into multiple waves
3. Fast Fourier Transform
4. Spectrogram - frequency of decomposed sound file over time


**Future Direction**

1. drag and drop applet - add song, predict genre (not just classical)