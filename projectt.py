from tkinter import *
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np
import pickle
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer #loading tfidf vector
from keras.models import Sequential
from keras.layers import Dense, Flatten, Bidirectional, LSTM, RepeatVector, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint
import os
import pandas as pd
from string import punctuation
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
from nltk.stem import PorterStemmer
from sklearn.metrics import accuracy_score

main=tkinter.Tk()
main.title("Generating Synthetic Image text using Deep Learning")
main.geometry("1200x1200")

global X_train,X_test,Y_train,Y_test,tfidf_vectorizer,sc
global model
global filename
global x,y,dataset

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

#define function to clean text by removing stop words other special symbols
def cleanText(doc):
    token = doc.split()
    tabel = str.maketrans('','',punctutaion)
    tokens = [w.translate(tabel) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word)>1]
    tokens = [ps.stem(token)for token in tokens]
    tokens = [lemmatizer.lemmatize(token)for token in tokens]
    tokens = ' '.join(tokens)
    return token
def uploadDataset():
    global filename, dataset
    text.delete('1.0',END)
    filename = filedialog.askdirectory(initialdir=".")
    text.insert(END,str(filename)+"DATASET LOADED\n\n")
    pathlabel.config(text=str(filename)+"DATASET LOADED\n\n")
    dataset = pd.read_csv("Dataset/text.txt")
    dataset = dataset.values
    text.insert(END,"Total Text & images loaded from datset:"+str(dataset.shape[0]+"\n\n"))

def preprocessDataset():
    global filename, dataset,x,y,tfidf_vectoriaer,sc
    global X_train,Y_train,X_test,y_test
    text.delete('1.0',END)
    if os.path.exsits("model/x.npy"):
        X = np.load("model/x.npy")
        y = np.load("model/y.npy")
    else:
        x=[]
        y=[]
        dataset = pd.read_csv("Dataset/caption.txt",nrows=300)
        dataset = dataset.values
        for i in range(len(dataset)):
            image_name =data[i,0]
            features = cv2.imread("dataset/images/"+image_name)
            features = cv2.resize(features,(128,128))
            y.append(features)
            answer = dataset[i,1]
            answer = answer.lower().strip()
            answer = cleanText(answer)
            X.append(answer)
        X = np.asarray(X)
        Y = np.asarray(y)
        tfidf_vectorizer = TfifVectorizer(stop_words=stop_words, use_idf=True, smooth_idf=false,norm = None)
        X=tfidf_vectorizer.fit_transform(X).toarray()
        data = X
        sc = StandardScaler()
        X = sc.fit_transform(X)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1, 1))
# The line below flattens the image data. The extra parentheses around the second argument are in the original image.
        Y = np.reshape(Y, (Y.shape[0], (Y.shape[1] * Y.shape[2] * Y.shape[3]))) 
        Y = Y.astype('float32')
        Y = Y/255
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train
        text.insert(END, "Training Text & Image Generation Processing Completed\n\n")
        text.insert(END, "Normalized Text Vector : \n\n")
        text.insert(END, str(data))
def trainDL():
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test, model
    model = Sequential()
    
    #creating layer to extract features from images
    model.add(Conv2D(32, (1, 1), activation='relu', input_shape=(X.shape[1], X.shape[2], X.shape[3])))
    model.add(MaxPooling2D((1, 1)))
    
    #adding another layer
    model.add(Conv2D(64, (1, 1), activation='relu'))
    model.add(MaxPooling2D((1, 1)))
    model.add(Conv2D(128, (1, 1), activation='relu'))
    model.add(MaxPooling2D((1, 1)))
    model.add(Flatten())
    model.add(RepeatVector(2))
    
    #adding bidirectional + LSTM layer to train TEXT features
    model.add(Bidirectional(LSTM(128, activation = 'relu')))
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(Y.shape[1], activation='sigmoid'))
    
    # Compile and train the model.
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    if os.path.exists("model/cnn_weights.hdf5") == False:
        import os
import pickle
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import RepeatVector, Bidirectional, LSTM, Dropout, Dense

# Assuming 'model', 'X_train', 'y_train', 'X_test', 'y_test', 'Y', and 'text'
# are already defined in the preceding code.

model.add(RepeatVector(2))
#adding bidirectional + LSTM layer to train TEXT features
model.add(Bidirectional(LSTM(128, activation = 'relu')))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(Y.shape[1], activation='sigmoid'))

# Compile and train the model
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

if os.path.exists("model/cnn_weights.hdf5") == False:
    # If weights file doesn't exist, train the model
    check_point = ModelCheckpoint(filepath='model/cnn_weights.hdf5', verbose=1, save_best_only=True)
    hist = model.fit(X_train, y_train, batch_size=16, epochs=15, validation_data=(X_test, y_test))
    
    # Save the training history
    with open('model/cnn_history.pkl', 'wb') as f:
        pickle.dump(hist.history, f)
else:
    # If weights file exists, load the weights and history
    model.load_weights("model/cnn_weights.hdf5")
    
    with open('model/cnn_history.pkl', 'rb') as f:
        data = pickle.load(f)
        
    print(data['accuracy'])
    accuracy_value = data['accuracy'][14] # Get accuracy from the last epoch
    text.insert(END, "Model Accuracy : " + str(accuracy_value))
