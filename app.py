import streamlit as st
import pickle
import sklearn

def text_preprocessing(text):
    from nltk.tokenize import word_tokenize
    from nltk.stem.porter import PorterStemmer
    import string
    from nltk.corpus import stopwords
    text = text.lower()
    
    for punc in string.punctuation:
        text = text.replace(punc,'')
    
    text_list = word_tokenize(text)
    
    for word in text_list:
        if word in stopwords.words('english'):
            text_list.remove(word)
    ps = PorterStemmer()
    
    for index in range(len(text_list)):
        text_list[index] = ps.stem(text_list[index])
        
    return ' '.join(text_list)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = text_preprocessing(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")