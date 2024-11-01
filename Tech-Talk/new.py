import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = []
classes = []
doc = []
ignorelet = ['?', '!', '.', ',']

# Tokenizing words and preparing classes
for intent in intents['intents']:
    for patterns in intent['patterns']:
        wordList = nltk.word_tokenize(patterns)
        words.extend(wordList)
        doc.append((wordList, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and sort words, and ignore punctuation
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignorelet]
words = sorted(set(words))

# Sort classes
classes = sorted(set(classes))

# Save words and classes for later use
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Prepare training data
training = []
outputempty = [0] * len(classes)

for docs in doc:
    bag = []
    wordPatterns = docs[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)
    
    outputRow = list(outputempty)
    outputRow[classes.index(docs[1])] = 1
    training.append(bag + outputRow)

random.shuffle(training)
training = np.array(training)

# Split the features and labels
trainX = training[:, :len(words)]
trainY = training[:, len(words):]

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(trainY[0]), activation='softmax')
])

# Compile the model
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
hist = model.fit(np.array(trainX), np.array(trainY), epochs=200, batch_size=5, verbose=1)

# Save the model
model.save('Tech_Talk.h5', hist)

print("Executed")
