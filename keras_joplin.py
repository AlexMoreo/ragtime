import numpy as np
from keras import Input
from keras import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import *
from keras import optimizers
# from parse_ragtime import *



X = np.load('np/sunflower.npy')

MAX_SEQUENCE_LENGTH = 16*8

def RNN(lstmsize=512):
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH-1,88), dtype='float32')
    layer = LSTM(lstmsize, return_sequences=True, dropout=0.5, recurrent_dropout=0.5)(sequence_input)
    layer=Dropout(0.5)(layer)
    layer = Dense(1024, activation='relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(88, activation='sigmoid')(layer)
    model = Model(inputs=sequence_input, outputs=layer)
    return model

def sample(X,nsamples):
    positions = np.random.randint(0,X.shape[0]-MAX_SEQUENCE_LENGTH, nsamples)
    S = np.zeros((nsamples,MAX_SEQUENCE_LENGTH,88))
    for i,p in enumerate(positions):
        S[i]=X[p:p+MAX_SEQUENCE_LENGTH]
    return S

optimizer='rmsprop'
model = RNN()
model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['acc'])


# earlystop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, mode='min')
# val= X[-MAX_SEQUENCE_LENGTH * 2:]
# val_x, val_y= val[:-1], val[1:]

train = X[:-MAX_SEQUENCE_LENGTH*4]
val = X[-MAX_SEQUENCE_LENGTH*4:]

train_sample=sample(train, 1000)
val_sample=sample(val, 4)
x_train= train_sample[:, :-1, :]
y_train= train_sample[:, 1:, :]
x_val = val_sample[:, :-1, :]
y_val = val_sample[:, 1:, :]
history = model.fit(x_train, y_train,
                    validation_data=(x_val, y_val),
                    epochs=100,
                    batch_size=100,
                    shuffle=True)
                    # callbacks=[earlystop])

ragtime = []
input = X[:MAX_SEQUENCE_LENGTH-1]
print('writing a ragtime!')
for i in range(64*16):
    rag = model.predict(input[np.newaxis,:,:])
    rag = rag[0,-1,:]
    #sample rag keyboard
    keyboard = 1.0*(rag>0.5) #this is not sampling
    # if keyboard.sum()==0:
    # keyboard=np.zeros(88)
    # rag = rag / np.sum(rag)
    # keyboard[np.random.choice(88, 3, replace=False, p=rag)]=1
    # keyboard[np.argmax(rag-keyboard)] = 1
    ragtime.append(keyboard)
    input = np.vstack((input[1:],keyboard))

ragtime=np.vstack(ragtime)
np.save('generated/sunflower.npy',ragtime)
# numpy2midi(ragtime, 'generated/sunflower.npy')
print(ragtime.shape)
print(ragtime)