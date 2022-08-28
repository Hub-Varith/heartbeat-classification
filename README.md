# Heartbeat Classification

## Objective

Classify heartbeat sounds obtained from a digital stethoscope into 3 classes: Normal, Murmur, and Extrastole (Extrasystole).

## Dataset

The dataset is part of a Kaggle dataset called Heartbeat Sounds.

[https://www.kaggle.com/datasets/kinguistics/heartbeat-sounds?select=set_b.csv](https://www.kaggle.com/datasets/kinguistics/heartbeat-sounds?select=set_b.csv)

In this challenge, we will use the set_b dataset which contains data for heart beats collected from a clinical trial in hospitals using a digital stethoscope. The `set_b.csv` contains metadata and labels for the data.

## Methodology and Data Visualization

### Labels

We first visualize the dataset and the labels. It was found that the dataset had the following flaws:

1. The murmur class has 3x less data than the normal class
2. Extrastole (Extrasystole) is 6x less than the normal class

![class_value_counts.svg](Heartbeat%20Classification%20547a5bfeae644a9db04fe93a7807af48/class_value_counts.svg)

It can then be concluded that an upsample of extrasystole and murmur is required in order for the model to train with an equivalent number of classes and therefore maximize the performance.

Next, we’ll have a look at the `set_b.csv` file. 

![Screen Shot 2565-08-28 at 17.43.14.png](Heartbeat%20Classification%20547a5bfeae644a9db04fe93a7807af48/Screen_Shot_2565-08-28_at_17.43.14.png)

The audio file names are not the same as the *fname* labels. Therefore, we’ll have to delete the part which is not the same. Furtherore, there are *sublabel* features which are irrelevant to our training, therefore, we’ll drop *sublabel* out. We’ll also convert *label* into numerical values, and one-hot encode.

---

### Sound Analysis

Raw signal:

![soundwave_comparison.svg](Heartbeat%20Classification%20547a5bfeae644a9db04fe93a7807af48/soundwave_comparison.svg)

We then visualize the different types of sounds. It can be seen that the normal heartbeat has a very clear signal and peaked amplitudes per time interval are seen in a constant pattern, whereas  some peaked amplitudes are off-patterned for the murmur sound. For extrasystole, the wave characteristics are similar to that of the normal heartbeat, therefore, we can only compare the normal sound wave to the extrasystole sound wave via audio comparison.

Normal STFT vs Log STFT vs MFCC:

![stft_mfcc_rep.svg](Heartbeat%20Classification%20547a5bfeae644a9db04fe93a7807af48/stft_mfcc_rep.svg)

In this challenge, we used MFCC as our input feature which summarizes the frequency distribution across the window size, which makes it possible to visualize both frequency and time characteristics of the sound.

---

### Model Architecture

The model used relatively simple. We used a Conv2d with 32 kernels and a size of 3*3. Then we go through Batch Normalization, Maxpooloing and Dropout, to prevent overfitting. Lastly, we flatten the layers and have a 512 node Dense layer and an output of 3 Nodes, using the Softmax activation function.

```python
model = Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3), activation='relu', input_shape=(13, 646, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', patience=10)])
model.summary()
```

Hyperparameters:

epochs: 50

batch_size: 32

optimizer: Adam

loss: categorical_crossentropy

---

## Results

### Model accuracy & Loss

![model_accuracy.svg](Heartbeat%20Classification%20547a5bfeae644a9db04fe93a7807af48/model_accuracy.svg)

![model_loss.svg](Heartbeat%20Classification%20547a5bfeae644a9db04fe93a7807af48/model_loss.svg)

It was found that the model accuracy, both for the training and validation set, eventually met at a value of around 0.9. The loss dramatically went down at the 3rd epoch very close to zero. The train and validation graphs aligns perfectly for the loss graph, therefore suggesting that the model does not overfit.

### Confusion Matrix

![confusion_matrix.svg](Heartbeat%20Classification%20547a5bfeae644a9db04fe93a7807af48/confusion_matrix.svg)

It was found that there were lots of true positives and a few prediction errors.

