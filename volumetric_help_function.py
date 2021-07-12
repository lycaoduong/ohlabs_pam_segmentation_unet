import numpy as np
import cv2
import itk


def img_2_npy(img_path, num_bscan):
    data = []
    for i in range(num_bscan):
        print(i)
        img = cv2.imread(img_path + "bscan_%s.png"%i)
        img = img[:,:,0]
        data.append(img)
    data = np.array(data)
    return data

def npy_2_nrrd(npy_file, save_path):
    image = itk.GetImageFromArray(npy_file.astype(np.uint8))
    itk.imwrite(image, save_path)


def cscan_reconstruct(npy_data):
    cscan = []
    num_bscan = npy_data.shape[0]
    for b in range(num_bscan):
        cscan.append(np.amax(npy_data[b, :, :], axis=0))
    cscan = np.array(cscan)
    return cscan