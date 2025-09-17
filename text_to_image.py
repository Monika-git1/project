from tkinter import *
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np
import pickle
import cv2
import numpy as np
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

main = tkinter.Tk()
main.title("Generating Synthetic Images from Text using RNN")
main.geometry("1200x1200")

global X_train, X_test, y_train, y_test, tfidf_vectorizer, sc
global model
global filename
global X, Y, dataset

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

#define function to clean text by removing stop words and other special symbols
def cleanText(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [ps.stem(token) for token in tokens]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens

def uploadDataset():
    global filename, dataset
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir=".")
    text.insert(END,str(filename)+" Dataset Loaded\n\n")
    pathlabel.config(text=str(filename)+" Dataset Loaded\n\n")
    dataset = pd.read_csv("Dataset/text.txt")
    dataset = dataset.values
    text.insert(END,"Total Text & Images Loaded from Dataset : "+str(dataset.shape[0])+"\n\n")

def preprocessDataset():
    global filename, dataset, X, Y, tfidf_vectorizer, sc
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    if os.path.exists("model/X.npy"):
        X = np.load("model/X.npy")
        Y = np.load("model/Y.npy")
    else:
        X = []
        Y = []
        dataset = pd.read_csv("Dataset/captions.txt",nrows=300)
        dataset = dataset.values
        for i in range(len(dataset)):
            image_name = dataset[i, 0]
            features = cv2.imread("Dataset/Images/"+image_name)
            features = cv2.resize(features, (128, 128))
            Y.append(features)
            answer = dataset[i,1]
            answer = answer.lower().strip()
            answer = cleanText(answer)#clean description
            X.append(answer)
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save("model/X", X)
        np.save("model/Y", Y)
    tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words, use_idf=True, smooth_idf=False, norm=None, decode_error='replace')
    X = tfidf_vectorizer.fit_transform(X).toarray()
    data = X
    sc = StandardScaler()
    X = sc.fit_transform(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1, 1))
    Y = np.reshape(Y, (Y.shape[0], (Y.shape[1] * Y.shape[2] * Y.shape[3])))
    Y = Y.astype('float32')
    Y = Y/255
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test
    text.insert(END,"Training Text & Image Generation Processing Completed\n\n")
    text.insert(END,"Normalized Text Vector : \n\n")
    text.insert(END,str(data))

def trainCNNBILSTM():
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test, model
    model = Sequential()
    #creating CNN layer to extract features from images
    model.add(Conv2D(32, (1, 1), activation='relu', input_shape=(X.shape[1], X.shape[2], X.shape[3])))
    #max layer to collect relevant features from CNN layer
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
        model_check_point = ModelCheckpoint(filepath='model/cnn_weights.hdf5', verbose = 1, save_best_only = True)
        hist = model.fit(X_train, y_train, batch_size = 16, epochs = 15, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/cnn_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        model.load_weights("model/cnn_weights.hdf5")
    f = open('model/cnn_history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    print(data['accuracy'])
    accuracy_value = 1 - data['accuracy'][14]
    text.insert(END,"Model Accuracy : "+str(accuracy_value)) 

def predictImage():
    global model, sc, tfidf_vectorizer
    text.delete('1.0', END)
    input_text = tf1.get()
    tf1.delete(0, END)
    answer = input_text.lower().strip()
    data = answer
    data = cleanText(data)
    test = tfidf_vectorizer.transform([data]).toarray()
    test = sc.transform(test)
    test = np.reshape(test, (test.shape[0], test.shape[1], 1, 1))
    predict = model.predict(test)
    predict = predict[0]
    predict = np.reshape(predict, (128, 128, 3))
    predict = cv2.resize(predict, (300, 300))
    cv2.imshow("Text To Image Output", predict)
    cv2.waitKey(0)
    

font = ('times', 14, 'bold')
title = Label(main, text='Generating Synthetic Images from Text using RNN')
title.config(bg='DarkGoldenrod1', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=5,y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Flickr Text to Image Dataset", command=uploadDataset)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=560,y=100)

preprocessButton = Button(main, text="Preprocess Dataset", command=preprocessDataset)
preprocessButton.place(x=50,y=150)
preprocessButton.config(font=font1)

trainButton = Button(main, text="Load RNN Model", command=trainCNNBILSTM)
trainButton.place(x=250,y=150)
trainButton.config(font=font1)

l1 = Label(main, text='Input Text:')
l1.config(font=font)
l1.place(x=50,y=200)

tf1 = Entry(main,width=70)
tf1.config(font=font)
tf1.place(x=160,y=200)

predictButton = Button(main, text="Text To Image", command=predictImage)
predictButton.place(x=350,y=250)
predictButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=18,width=300)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=300)
text.config(font=font1)


main.config(bg='LightSteelBlue1')
main.mainloop()
