import random
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

def extract_random(full_imgs,full_masks, patch_h,patch_w, N_patches):
    if (N_patches%full_imgs.shape[0] != 0):
        print("Program exit: please enter a multiple of num_image train")
        print("N_patches: ", N_patches)
        print("Total images train: ", full_imgs.shape[0])
        exit()
    assert (len(full_imgs.shape)==3 and len(full_masks.shape)==3)  #3D arrays
    assert (full_imgs.shape[1] == full_masks.shape[1] and full_imgs.shape[2] == full_masks.shape[2])
    patches = np.empty((N_patches,patch_h,patch_w))
    patches_masks = np.empty((N_patches,patch_h,patch_w))
    img_h = full_imgs.shape[1]  #height of the full image
    img_w = full_imgs.shape[2] #width of the full image
    # (0,0) in the center of the image
    patch_per_img = int(N_patches/full_imgs.shape[0])  #N_patches equally divided in the full images
    print("patches per full image: " +str(patch_per_img))
    iter_tot = 0   #iter over the total numbe rof patches (N_patches)
    for i in range(full_imgs.shape[0]):  #loop over the full images
        k=0
        while k <patch_per_img:
            x_center = random.randint(0+int(patch_w/2),img_w-int(patch_w/2))
            # print "x_center " +str(x_center)
            y_center = random.randint(0+int(patch_h/2),img_h-int(patch_h/2))

            patch = full_imgs[i,y_center-int(patch_h/2):y_center+int(patch_h/2),x_center-int(patch_w/2):x_center+int(patch_w/2)]
            patch_mask = full_masks[i,y_center-int(patch_h/2):y_center+int(patch_h/2),x_center-int(patch_w/2):x_center+int(patch_w/2)]
            patches[iter_tot]=patch
            patches_masks[iter_tot]=patch_mask
            iter_tot +=1   #total
            k+=1  #per full_img
    return patches, patches_masks

def extract_ordered_overlap(full_imgs, patch_h, patch_w,stride_h,stride_w):
    img_h = full_imgs.shape[1]
    img_w = full_imgs.shape[2]
    img_class = full_imgs.shape[3]
    assert ((img_h - patch_h) % stride_h == 0 and (img_w - patch_w) % stride_w == 0)
    N_patches_img = ((img_h - patch_h) // stride_h + 1) * ((img_w - patch_w) // stride_w + 1)
    N_patches_img_total = N_patches_img*full_imgs.shape[0]
    print("Num_img_extract: ", N_patches_img_total)
    patches = np.empty((N_patches_img_total,  patch_h, patch_w, img_class))
    iter_tot = 0
    for i in range(full_imgs.shape[0]):  #loop over the full images
        for h in range((img_h-patch_h)//stride_h+1):
            for w in range((img_w-patch_w)//stride_w+1):
                patch = full_imgs[i,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]
                patches[iter_tot]=patch
                iter_tot +=1   #total
    assert (iter_tot==N_patches_img_total)
    return patches  #array with all the full_imgs divided in patches

def recompose_overlap(preds, img_h, img_w, stride_h, stride_w):
    patch_h = preds.shape[1]
    patch_w = preds.shape[2]
    n_class = preds.shape[3]
    N_patches_h = (img_h-patch_h)//stride_h+1
    N_patches_w = (img_w-patch_w)//stride_w+1
    N_patches_img = N_patches_h * N_patches_w
    # print("N_patches_h: " +str(N_patches_h))
    # print("N_patches_w: " +str(N_patches_w))
    # print("N_patches_img: " +str(N_patches_img))
    N_full_imgs = preds.shape[0]//N_patches_img
    # print("According to the dimension inserted, there are " + str(N_full_imgs) + " full images (of " + str(img_h) + "x" + str(img_w) + " each)")
    full_prob = np.zeros((N_full_imgs, img_h, img_w, n_class))
    full_sum = np.zeros((N_full_imgs, img_h, img_w, n_class))
    k = 0
    for i in range(N_full_imgs):
        for h in range(N_patches_h):
            for w in range(N_patches_w):
                full_prob[i, h*stride_h:(h*stride_h+patch_h), w*stride_w:(w*stride_w+patch_w), :] += preds[k]
                full_sum[i, h * stride_h:(h * stride_h + patch_h), w * stride_w:(w * stride_w + patch_w), :] += 1
                k+=1
    assert (k==preds.shape[0])
    final_img_pred = full_prob/full_sum
    # print(final_img_pred.shape)
    return final_img_pred

def find_submultiple(n):
    sub = []
    for i in range(1, n+1):
        if(n%i==0):
            sub.append(i)
    sub = np.array(sub)
    return sub

def find_stride(full_size_h, full_size_w, patch_h, patch_w):
    stride_h = find_submultiple(full_size_h-patch_h)
    stride_w = find_submultiple(full_size_w-patch_w)
    stride_h = stride_h[stride_h<=patch_h]
    stride_w = stride_w[stride_w<=patch_w]
    stride_h = np.max(stride_h)
    stride_w = np.max(stride_w)
    return  stride_h, stride_w

def one_hot_encoder(inputs):
    o = np.zeros((inputs.shape[0], inputs.shape[1], inputs.max()+1))
    layer_idx = np.arange(inputs.shape[0]).reshape(inputs.shape[0], 1)
    component_idx = np.tile(np.arange(inputs.shape[1]), (inputs.shape[0], 1))
    o[layer_idx, component_idx, inputs] = 1
    return o
def one_hot_decoder(inputs):
    o = np.argmax(inputs, axis=2)
    return o

def data_augmentation(img_folder, mask_folder, gen_img, gen_mask, size = (1024, 1000)):
    datagen = ImageDataGenerator(rotation_range=90, width_shift_range=0.1, height_shift_range=0.1)
    maskgen = ImageDataGenerator(rotation_range=90, width_shift_range=0.1, height_shift_range=0.1)

    train_data = datagen.flow_from_directory(img_folder, target_size=size,
                                             batch_size=32, class_mode='binary', seed=2,
                                             save_to_dir=gen_img)
    mask_data = datagen.flow_from_directory(mask_folder, target_size=size,
                                            batch_size=32, class_mode='binary', seed=2,
                                            save_to_dir=gen_mask)
    for i in range(len(train_data)):
        train_data.next()
        mask_data.next()
# inp = np.array([[0,1,2,2,1],[2,2,0,2,0]])
#
# o = one_hot_encoder(inp)
# print(o)