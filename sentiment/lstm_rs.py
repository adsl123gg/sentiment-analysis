from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers
from sklearn.metrics import confusion_matrix
import pandas as pd
from functools import reduce

df_train = pd.read_csv('../datasets/sentiment_train.csv')
x_train = df_train['text'].values
y_train = df_train['label'].values

y_train[y_train == 'Positive'] = 1
y_train[y_train != 'Positive'] = 0

df_test = pd.read_csv('../datasets/sentiment_test.csv')
x_test = df_test['text'].values

tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(x_train)
xtrain = tokenizer.texts_to_sequences(x_train)
xtest = tokenizer.texts_to_sequences(x_test)

def maxWordLen(a, b):
    if a>b:
        return a
    else:
        return b

max_train = reduce(maxWordLen, map(len, xtrain))
max_test = reduce(maxWordLen, map(len, xtest))

vocab_size = len(tokenizer.word_index)+1

maxlen = max(max_train, max_test)
xtrain = pad_sequences(xtrain, padding='post', maxlen=maxlen)
xtest = pad_sequences(xtest, padding='post', maxlen=maxlen)


print(x_train[3])
print(xtrain[3])


embedding_dim=50
model=Sequential()
model.add(layers.Embedding(input_dim=vocab_size,
                           output_dim=embedding_dim,
                           input_length=maxlen))
model.add(layers.LSTM(units=50,return_sequences=True))
model.add(layers.LSTM(units=10))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(8))
model.add(layers.Dense(1, activation="sigmoid"))
model.compile(optimizer="adam", loss="binary_crossentropy",
              metrics=['accuracy'])
model.summary()
model.fit(xtrain,y_train, epochs=20, batch_size=16, verbose=False)

loss, acc = model.evaluate(xtrain, y_train, verbose=False)
print("Training Accuracy: ", acc.round(2))

ypred=model.predict(xtest)

ypred[ypred>0.5]=1
ypred[ypred<=0.5]=0

result=zip(x_test, ypred)
for i in result:
    print(i)