import numpy as np
import os
import matplotlib.pyplot as plt
from model import *
from keras.layers import Input, add, Multiply, Activation
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from datetime import datetime
import cv2
from model_help_function import *
from data_help_function import *
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Augmentation Data Generation

# img_folder = './data/train/hand/image'
# mask_folder = './data/train/hand/label'
# img_gen = './data/train/hand/gen_image'
# msk_gen = './data/train/hand/gen_label'
# data_augmentation(img_folder, mask_folder, img_gen, msk_gen)


patch_w = 256
patch_h = 256


# file_path = "./data/pam/foot/train/"
file_path = "./data/train/hand/"
xpath = os.path.join(file_path, "gen_image")
ypath = os.path.join(file_path, "gen_label")

data = os.listdir(xpath)
label = os.listdir(ypath)

img_h = 1024
img_w = 1000
num_class = 0

x_image = np.zeros((len(data), img_h, img_w, 1))
y_image = np.zeros((len(label), img_h, img_w, num_class+1))


for idx, imname in enumerate(data):
    print(os.path.join(xpath, imname))
    img = cv2.imread(os.path.join(xpath, imname))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray/255
    gray = np.expand_dims(gray, axis=2)
    x_image[idx, :, :, :] = gray

print(x_image.shape)

for idx, imname in enumerate(label):
    print(os.path.join(ypath, imname))
    img = cv2.imread(os.path.join(ypath, imname))
    img = img[:,:,0]/255
    gray = np.expand_dims(img, axis=2)
    # img = one_hot_encoder(img)
    # y_image[idx, :, :, :] = img
    y_image[idx, :, :, :] = gray

print(y_image.shape)

str_h, str_w = find_stride(img_h, img_w, patch_h, patch_w)
print("stride: ", str_h, str_w)

x_train = extract_ordered_overlap(x_image, patch_h, patch_w, str_h, str_w)
y_train = extract_ordered_overlap(y_image, patch_h, patch_w, str_h, str_w)
print("Xtrain: ", x_train.shape, y_train.shape)


# x_train = np.expand_dims(x_train, axis=3)
# y_train = np.expand_dims(y_train, axis=3)

inpt = Input(shape=(256, 256, 1))
print(inpt.shape)

model = unet(inpt, n_classes = num_class+1)
adam = Adam(lr=0.0001) #0.0001
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

# model = load_model('tu_new_foot.h5')

# checkpoint = ModelCheckpoint("best_weight_PAM.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max', period=1)
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint('./data/model_checkpoint/hand_unet_bestweight.h5', save_best_only=True, monitor='val_loss', mode='min', verbose=1, period=1)
# reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

result = model.fit(x_train, y_train, batch_size=4, epochs=20, validation_split=0.33, callbacks=[earlyStopping, mcp_save])
# result = model.fit(x_train, y_train, batch_size=4, epochs=2, validation_split=0.4)

acc_loss = []
acc_loss.append(result.history['accuracy'])
acc_loss.append(result.history['val_accuracy'])
acc_loss.append(result.history['loss'])
acc_loss.append(result.history['val_loss'])
acc_loss = np.array(acc_loss)
print(acc_loss.shape)
np.save("./data/model_checkpoint/acc_loss.npy", acc_loss)

# plt.plot(np.arange(len(result.history['accuracy'])), result.history['accuracy'], label='training')
# plt.plot(np.arange(len(result.history['val_accuracy'])), result.history['val_accuracy'], label='validation')
plt.plot(acc_loss[0,:], label='training')
plt.plot(acc_loss[1,:], label='validation')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# plt.plot(np.arange(len(result.history['loss'])), result.history['loss'], label='training')
# plt.plot(np.arange(len(result.history['val_loss'])), result.history['val_loss'], label='validation')
plt.plot(acc_loss[2,:], label='training')
plt.plot(acc_loss[3,:], label='validation')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


model.save('./data/model_checkpoint/hand_unet_lastweight.h5')

for i in range(4):
    output = model.predict(np.expand_dims(x_train[3+i, :, :, :], axis=0))
    # output[output>=0.5] = 1
    # output[output<0.5] = 0
    plt.figure()
    plt.subplot(131)
    plt.imshow(np.squeeze(output), cmap="gray")
    plt.subplot(132)
    plt.imshow(y_train[3+i, :, :, :], cmap="gray")
    plt.subplot(133)
    plt.imshow(x_train[3+i, :, :, 0], cmap="gray")
    plt.show()
