import streamlit as st
import pickle
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
import numpy as np


list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]



    
def preprocess_input(text):
    # Load the tokenizer
    with open('tokenizer.pickle', 'rb') as handle:
        loaded_tokenizer = pickle.load(handle)
    # Tokenize text
    tokenized_text = loaded_tokenizer.texts_to_sequences([text])
    # Pad sequences
    padded_text = pad_sequences(tokenized_text, maxlen=200)
    return padded_text



def predict_class(text):
    padded_text = preprocess_input(text)
    # Load model architecture
    with open("model.json", "r") as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)

    # Load model weights
    loaded_model.load_weights("model_weights.h5")
    
    return loaded_model.predict(padded_text) # Assuming prediction are a multiple values

def main(): 
    st.title("Negative Comments Predictor")
    html_temp = """
    <style>
       .stApp {
        color: white;
    }
    .prediction-box {
        background-color: #ffffff;
        padding: 10px;
        border-radius: 5px;
        margin-top: 20px;
    }
    .estimate {
        font-weight: bold;
    }
    .bound {
        margin-top: 5px;
    }
    </style>
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">House Price Predictor App </h2>
    </div>
    """

    text = st.text_input("Comment", "enter comment") 

    if st.button("Predict"): 
        
        prediction = predict_class(
            text
        )
        
        if np.max(np.where(prediction<=0.5,0,1)) != 1   :         
            st.success('your comment have been sent')
        else :
            st.success(f'your comment is {list_classes[np.argmax(prediction)]}')
        
        
        

if __name__ == '__main__': 
    # Run Streamlit app
    main()