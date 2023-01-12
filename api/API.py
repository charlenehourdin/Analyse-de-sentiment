import os
import streamlit as st
#from pyngrok import ngrok
import os 
import numpy as np
import json
import pickle
from time import time

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
#from transformers import BertTokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
import os
import streamlit as st

#model_path = os.path.join('/Users/charlenehourdin/Documents/Openclassrooms/Projet/p7/api')
    
# Charger le modèle
glove_model = load_model('GlOVE_LSTM_lem.h5')

with open('tokenizer_cl.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Traiter la demande au service
def run(data):
  try:
      # Choisissez la propriété text de la requête JSON
      # Détails JSON attendus {"text": "un texte à évaluer pour le sentiment"}
      data = json.loads(data)
      prediction = predict(data['text'])
      return prediction
  except Exception as e:
      error = str(e)
      return error

# Déterminer le sentiment à partir du score
NEGATIVE = 'NEGATIF'
POSITIVE = 'POSITIF'
def decode_sentiment(score):
  if score < 0.5:
    label = NEGATIVE
  elif score == 0.5:
    label = POSITIVE
    #return NEGATIVE if score < 0.5 else POSITIVE
    return label


# Prédire le sentiment à l'aide du modèle
SEQUENCE_LENGTH = 36
def predict(text):
  start = time()
    
  # Tokenizer le texte
  x_test = pad_sequences(tokenizer.texts_to_sequences([text]),
                          maxlen=SEQUENCE_LENGTH)
  
  # Predire
  score = glove_model.predict([x_test])[0]
  
  # Decoder le sentiment
  label = decode_sentiment(score)
  elapsed_time =  time()-start
  #return float(score)
  return label, score, elapsed_time
  #return {'label': label, 'score': float(score),
      #elapsed_time': time()-start}

from PIL import Image
image = Image.open('air_paradis.png')
st.image(image)
st.markdown('---')

st.markdown("<h1 style='text-align: center; color: DarkRed;'>Analyse de sentiment</h1>", unsafe_allow_html=True)
ip = st.text_input('Entrer votre commentaire:')
#op = pad_sequences(ip, maxlen=128, padding='post')
label, score, elapsed_time = predict(ip)
if st.button('Envoyer'):
  #st.title(label)
  #st.markdown("<h4 style='text-align: center; color: DarkRed;'>- Inferieur à 50% = Negatif</h4>", unsafe_allow_html=True)
  #st.markdown("<h4 style='text-align: center; color: DarkGreen;'>- Supérieur à 50% = Positif</h4>", unsafe_allow_html=True)

  import plotly.graph_objects as go
  import numpy as np

  plot_bgcolor = "#FFFFFF"
  quadrant_colors = [plot_bgcolor, "#2bad4e", "#f25829"] 
  quadrant_text = ["", "<b>Positif</b>", "<b>Négatif</b>"]
  n_quadrants = len(quadrant_colors) - 1

  current_value = float(score)
  min_value = 0
  max_value = 0.99
  hand_length = np.sqrt(2) / 7
  hand_angle = np.pi * (1 - (max(min_value, min(max_value, current_value)) - min_value) / (max_value - min_value))

  fig = go.Figure(
      data=[
          go.Pie(
              values=[0.5] + (np.ones(n_quadrants) / 2 / n_quadrants).tolist(),
              rotation=90,
              hole=0.5,
              marker_colors=quadrant_colors,
              text=quadrant_text,
              textinfo="text",
              hoverinfo="skip",
          ),
      ],
      layout=go.Layout(
          showlegend=False,
          margin=dict(b=0,t=10,l=10,r=10),
          width=450,
          height=450,
          paper_bgcolor=plot_bgcolor,
          annotations=[
              go.layout.Annotation(
                  text=f"<b>Score:</b><br>{round(current_value*100)}",
                  x=0.5, xanchor="center", xref="paper",
                  y=0.25, yanchor="bottom", yref="paper",
                  showarrow=False,
                  font=dict(size=20, color="red")
              )
          ],
          shapes=[
            go.layout.Shape(
                type="circle",
                x0=0.48, x1=0.52,
                y0=0.48, y1=0.52,
                fillcolor="#333",
                line_color="#333",
            ),
            go.layout.Shape(
                type="line",
                x0=0.5, x1=0.5 + hand_length * np.cos(hand_angle),
                y0=0.5, y1=0.5 + hand_length * np.sin(hand_angle),
                line=dict(color="#333", width=4)
              )
          ]
      )
  )
  #fig.show()
  st.plotly_chart(fig)