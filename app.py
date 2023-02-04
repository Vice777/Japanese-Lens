from logging import exception
from threading import excepthook
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from PIL import Image
import time

from keras.applications import Xception
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tqdm.notebook import tqdm
tqdm().pandas()

st.set_option('deprecation.showPyplotGlobalUse', False)
st.title('Image Caption')
st.header('GIve you image a Caption')

def extract_features(directory):
        model = Xception( include_top=False, pooling='avg' )
        features = {}
        for img in tqdm(os.listdir(directory)):
            filename = directory + "/" + img
            image = Image.open(filename)
            image = image.resize((299,299))
            image = np.expand_dims(image, axis=0)
            #image = preprocess_input(image)
            image = image/127.5
            image = image - 1.0

            feature = model.predict(image)
            features[img] = feature
        return features

#2048 feature vector
dataset_images = "./Flickr8k_Dataset/Flicker8k_Dataset/"
features = pickle.load(open("features.p","rb"))


## Testing

def extract_features(filename, model):
        try:
            image = Image.open(filename)

        except:
            print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
        image = image.resize((299,299))
        image = np.array(image)
        # for images that has 4 channels, we convert them into 3 channels
        if image.shape[2] == 4: 
            image = image[..., :3]
        image = np.expand_dims(image, axis=0)
        image = image/127.5
        image = image - 1.0
        feature = model.predict(image)
        return feature

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
         if index == integer:
             return word
    return None

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text


path = './Flickr8k_Dataset/Flicker8k_Dataset/107318069_e9f2ef32de.jpg'


nav_choice = st.sidebar.radio('Navigation', ['Home'], index=0)

if nav_choice == 'Home':

    st.success('Here, for the purpose of Image Caption, model use CNN for Feature Extraction'
               'and LSTM for text dependencies ')

    st.info('Upload Image :')
    image_sample = st.file_uploader('Image', ['png', 'jpg'])
    if image_sample:
        try:
            max_length = 32
            tokenizer = pickle.load(open("tokenizer.p","rb"))
            model = load_model('models/model_9.h5')
            xception_model = Xception(include_top=False, pooling="avg")

            photo = extract_features(image_sample, xception_model)
            img = Image.open(image_sample)

            description = generate_desc(model, tokenizer, photo, max_length)
            
            plt.imshow(img)
            plt.show()
            st.pyplot()

            load = st.progress(0)
            for i in range(100):
                time.sleep(0.00001)
                load.progress(i + 1)
            st.success(f'Image Caption : {description}')
        
        except Exception as e:
            print(e, type(e))
