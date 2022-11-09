from multiprocessing.connection import wait
import pandas as pd                                                          
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import speech_recognition as sr

import random, os
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play


import time


df = pd.read_csv('wellnesscsvfile.csv')
df.head()

sentences = ["This is an example sentence", "Each sentence is converted"]
model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
embeddings = model.encode(sentences)


#정리
df.head()
df['embedding'] = pd.Series([[]] * len(df))
df['embedding'] = df['User'].map(lambda x: list(model.encode(x)))
df.head()
df.to_csv('wellnesscsvfile.csv', index=False)


userchat = ""
embedding = model.encode(userchat)
df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
df.head()
encoded = df.loc[df['distance'].idxmax()]


# process D :telegram bot
import streamlit as st
import time

st.title('Hello, User. Talk to Nuru.')
st.markdown("[KIIID] 2022 Kyeong Hee University Industirual Design")

# process A : stt
def stt():
    global userchat
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        audio = recognizer.listen(source)
        try:
            userchat = recognizer.recognize_google(audio, language="ko")
            st.text('User : {}'.format(userchat))
        except:
            userchat = False

#process B : embeding and finding Nuru answer
def communicate():
    global Nuruchat
    embedding = model.encode(userchat)
    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    df.head()
    encoded = df.loc[df['distance'].idxmax()]
    Nuruchat = encoded['Nuru']
    st.text('Nuru : {}'.format(Nuruchat))

# process C : tts

os.makedirs('samples', exist_ok=True)
os.makedirs('result', exist_ok=True)

def tts():
    lang = 'ko'
    normal_frame_rate = 44100
    Nuru_voice = None

    for i, letter in enumerate(Nuruchat):
        if letter == ' ':
            new_sound = letter_sound._spawn(b'\x00' * (normal_frame_rate // 3), overrides={'frame_rate': normal_frame_rate})
        else:
            if not os.path.isfile('samples/%s.mp3' % letter):
                tts = gTTS(letter, lang=lang)
                tts.save('samples/%s.mp3' % letter)

            letter_sound = AudioSegment.from_mp3('samples/%s.mp3' % letter)
            raw = letter_sound.raw_data[8000:-8000]
            octaves = 1.8 + random.uniform(0.96, 1.15)
            frame_rate = int(letter_sound.frame_rate * (1.5 ** octaves))
            print('%s - octaves: %.2f, fr: %.d' % (letter, octaves, frame_rate))
            new_sound = letter_sound._spawn(raw, overrides={'frame_rate': frame_rate})

            new_sound = new_sound.set_frame_rate(normal_frame_rate)
            Nuru_voice = new_sound if Nuru_voice is None else Nuru_voice + new_sound

    Nuru_voice.export('result/%.mp3', format='mp3')
    play(Nuru_voice)

while userchat != '그만' :
    stt()
    if userchat == False :
        stt()
    else :
        communicate()
        tts()
        
        
