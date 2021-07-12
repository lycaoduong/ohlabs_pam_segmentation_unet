from keras.layers import Input, Conv2D, MaxPooling2D, add, Dropout, Conv2DTranspose, \
    UpSampling2D, concatenate, Activation, Add, Cropping2D, ZeroPadding2D, Multiply, Lambda, BatchNormalization, Convolution2D, Reshape
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Model
inpt = Input(shape=(256, 256, 1))

val1 = np.array([0.5])
val2 = np.array([0.5])
v1 = K.variable(value=val1)
v2 = K.variable(value=val2)


def unet(inputs=inpt, n_classes=1):
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    drop4 = Dropout(0.0)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    drop5 = Dropout(0.0)(conv5)



    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv10 = Conv2D(n_classes, 3, padding='same')(conv9)

    output = Activation('sigmoid')(conv10)
    model = Model(inputs, output)

    return model

def vgg_16_encoder(inputs=inpt):

    x = Conv2D(64, (3, 3), activation='relu', padding='same',
               name='block1_conv1', data_format='channels_last')(inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same',
               name='block1_conv2', data_format='channels_last')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool',
                     data_format='channels_last')(x)

    x = (BatchNormalization())(x)
    f1 = x
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same',
               name='block2_conv1', data_format='channels_last')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same',
               name='block2_conv2', data_format='channels_last')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool',
                     data_format='channels_last')(x)

    x = (BatchNormalization())(x)
    f2 = x

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
               name='block3_conv1', data_format='channels_last')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
               name='block3_conv2', data_format='channels_last')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
               name='block3_conv3', data_format='channels_last')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool',
                     data_format='channels_last')(x)

    x = (BatchNormalization())(x)
    f3 = x

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block4_conv1', data_format='channels_last')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block4_conv2', data_format='channels_last')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block4_conv3', data_format='channels_last')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool',
                     data_format='channels_last')(x)

    x = (BatchNormalization())(x)
    f4 = x

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block5_conv1', data_format='channels_last')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block5_conv2', data_format='channels_last')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block5_conv3', data_format='channels_last')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool',
                     data_format='channels_last')(x)

    x = (BatchNormalization())(x)
    f5 = x

    return inputs, [f1, f2, f3, f4, f5]

def crop(o1, o2):
    output_height2 = o2.shape[1]
    output_width2 = o2.shape[2]
    output_height1 = o1.shape[1]
    output_width1 = o1.shape[2]

    cx = abs(output_width1 - output_width2)
    cy = abs(output_height2 - output_height1)

    if output_width1 > output_width2:
        o1 = Cropping2D(cropping=((0, 0),  (0, cx)),
                        data_format='channels_last')(o1)
    else:
        o2 = Cropping2D(cropping=((0, 0),  (0, cx)),
                        data_format='channels_last')(o2)

    if output_height1 > output_height2:
        o1 = Cropping2D(cropping=((0, cy),  (0, 0)),
                        data_format='channels_last')(o1)
    else:
        o2 = Cropping2D(cropping=((0, cy),  (0, 0)),
                        data_format='channels_last')(o2)

    return o1, o2

def fcn_8(inputs=inpt, n_classes=1):
    img_input, levels = vgg_16_encoder(inputs)
    [f1, f2, f3, f4, f5] = levels

    o = f5

    o = (Conv2D(1024, (7, 7), activation='relu',
                padding='same', data_format='channels_last'))(o)
    o = Dropout(0.5)(o)
    o = (Conv2D(1024, (1, 1), activation='relu',
                padding='same', data_format='channels_last'))(o)
    o = Dropout(0.5)(o)

    o = (Conv2D(n_classes, (1, 1), kernel_initializer='he_normal',
                data_format='channels_last'))(o)
    o = Conv2DTranspose(n_classes, kernel_size=(4, 4), strides=(
        2, 2), use_bias=False, data_format='channels_last')(o)

    o = Cropping2D(((1, 1), (1, 1)))(o)

    o2 = f4
    o2 = (Conv2D(n_classes, (1, 1), kernel_initializer='he_normal',
                 data_format='channels_last'))(o2)

    # o, o2 = crop(o, o2)

    o = Add()([o, o2])

    o = Conv2DTranspose(n_classes, kernel_size=(4, 4), strides=(
        2, 2), use_bias=False, data_format='channels_last')(o)

    o = Cropping2D(((1, 1), (1, 1)))(o)

    o2 = f3
    o2 = (Conv2D(n_classes, (1, 1), kernel_initializer='he_normal',
                 data_format='channels_last'))(o2)
    # o2, o = crop(o2, o)
    o = Add(name="seg_feats")([o2, o])

    o = Conv2DTranspose(n_classes, kernel_size=(16, 16), strides=(
        8, 8), use_bias=False, data_format='channels_last')(o)

    o = Cropping2D(((4, 4), (4, 4)))(o)

    output = Activation('sigmoid')(o)
    model = Model(inputs, output)
    return model

def segnet(inputs=inpt, n_classes=1, level=5):
    img_input, levels = vgg_16_encoder(inputs)
    feat = levels[level-1]
    o = feat

    for _ in range(level-2):
        o = (UpSampling2D((2, 2), data_format='channels_last'))(o)
        o = (ZeroPadding2D((1, 1), data_format='channels_last'))(o)
        o = (Conv2D(512, (3, 3), padding='valid',
             data_format='channels_last'))(o)
        o = (BatchNormalization())(o)

    o = (ZeroPadding2D((1, 1), data_format='channels_last'))(o)
    o = (Conv2D(256, (3, 3), padding='valid', data_format='channels_last'))(o)
    # o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format='channels_last'))(o)
    o = (ZeroPadding2D((1, 1), data_format='channels_last'))(o)
    o = (Conv2D(128, (3, 3), padding='valid', data_format='channels_last'))(o)
    o = (BatchNormalization())(o)


    o = (UpSampling2D((2, 2), data_format='channels_last'))(o)
    o = (ZeroPadding2D((1, 1), data_format='channels_last'))(o)
    o = (Conv2D(64, (3, 3), padding='valid', data_format='channels_last', name="seg_feats"))(o)
    o = (BatchNormalization())(o)

    o = Conv2D(n_classes, (3, 3), padding='same',
               data_format='channels_last')(o)

    output = Activation('sigmoid')(o)
    model = Model(inputs, output)
    return model

# inpt = Input(shape=(256, 256, 1))
# print(inpt.shape)
#
# model = segnet(inpt, n_classes=1, level=5)

