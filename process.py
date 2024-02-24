import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import librosa
from pydub import AudioSegment

def process(filepath):
  saved_model_dir = 'model'
  model = tf.saved_model.load(saved_model_dir)
  classes =[  "ambulance" ,  "firetruck" ,  "traffic"   ]
  waveform , sr = librosa.load(filepath)#, sr=16000
  amplitude=np.max(np.abs(waveform))
  waveform=waveform[:16000]
  zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)
  waveform= tf.concat([zero_padding, waveform],0)
  inp = tf.constant( np.array([waveform]) , dtype='float32'  )
  class_scores = model( inp )[0].numpy()
  [  "ambulance" ,  "firetruck" ,  "traffic"   ]
  if classes[  class_scores.argmax()]=='ambulance' or classes[  class_scores.argmax()]=='firetruck':
    return [1,amplitude]
  else:
    return [0,0]

print(process('model/ambulance5.wav'))