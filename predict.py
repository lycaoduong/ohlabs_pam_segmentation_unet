import numpy as np
import os
from keras.models import load_model
from datetime import datetime
import cv2
from model_help_function import *
from data_help_function import *
import matplotlib.pyplot as plt
import cmapy

# model = load_model('best_skin_vessel.h5')
model = load_model('./data/model_checkpoint/hand_unet_bestweight.h5')

patch_w = 256
patch_h = 256



def predict_all_Bscan(img_path, save_path, multi_class=False):
    # img_path = os.path.join(img_path, img_folname)
    data = os.listdir(img_path)
    for idx, imname in enumerate(data):
        print(os.path.join(img_path, imname))
        img = cv2.imread(os.path.join(img_path, imname))
        gray = img[:,:,0]
        gray = gray / 255
        gray = np.expand_dims(gray, axis=2)
        x_image = gray
        # x_image = np.zeros((512, 2816))
        # x_image[0:300, 0:2600] = gray
        h = x_image.shape[0]
        w = x_image.shape[1]
        x_image = np.expand_dims(x_image, axis=0)
        str_h, str_w = find_stride(h, w, patch_h, patch_w)
        patches = extract_ordered_overlap(x_image, patch_h, patch_w, str_h, str_w)
        predict = model.predict(patches, verbose=1, batch_size = 4)
        img_predict = recompose_overlap(predict, h, w, str_h, str_w)

        if multi_class==True:
            img_predict[img_predict>=0.5] = 1
            img_predict[img_predict<0.5] = 0
            img_predict = one_hot_decoder(np.squeeze(img_predict, axis=0)) * 255/2
            img_predict = img_predict.astype(np.uint8)
        else:
            img_predict = np.squeeze(img_predict, axis=0)*255
            img_predict = img_predict.astype(np.uint8)

        img_predict = cv2.applyColorMap(img_predict, cv2.COLORMAP_VIRIDIS)
        cv2.imwrite(save_path + imname, img_predict)

def predict_from_img(img_path, multi_class=False):
    img = cv2.imread(img_path)
    gray = img[:, :, 0]
    gray = gray / 255
    gray = np.expand_dims(gray, axis=2)
    x_image = gray
    # x_image = np.zeros((512, 2816))
    # x_image[0:300, 0:2600] = gray
    h = x_image.shape[0]
    w = x_image.shape[1]
    x_image = np.expand_dims(x_image, axis=0)
    str_h, str_w = find_stride(h, w, patch_h, patch_w)
    patches = extract_ordered_overlap(x_image, patch_h, patch_w, str_h, str_w)
    predict = model.predict(patches, verbose=1, batch_size=4)
    img_predict = recompose_overlap(predict, h, w, str_h, str_w)
    if multi_class == True:
        img_predict[img_predict >= 0.5] = 1
        img_predict[img_predict < 0.5] = 0
        img_predict = one_hot_decoder(np.squeeze(img_predict, axis=0)) * 255 / 2
        img_predict = img_predict.astype(np.uint8)
    else:
        img_predict = np.squeeze(img_predict, axis=0) * 255

    plt.figure()
    plt.imshow(img_predict, cmap="viridis")
    plt.show()

# imgp = "./data/test/hand/image/"
# save = "./data/test/hand/predict/"
# predict_all_Bscan(imgp, save)

# imgp = "./data/test/hand/image/bscan_195.png"
# predict_from_img(imgp)

