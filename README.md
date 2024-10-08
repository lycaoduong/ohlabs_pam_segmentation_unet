### :star: This is the official repo for our paper **Full-view in vivo Skin and Blood Vessels Segmentation in Photoacoustic Imaging based on Deep Learning** published in Photoacoustics 2022 :star:


<p align="center">
      <a href="https://doi.org/10.1016/j.pacs.2021.100310"><img alt="Google Scholar" src="https://img.shields.io/badge/DOI-10.1016/j.pacs.2021.100310-green?color=FF8000&logoColor=white"></a>
      <img alt="Python" src="https://img.shields.io/badge/Python%20-%2314354C.svg?style=flat-square&logo=python&logoColor=white" />
      <img alt="Keras" src="https://img.shields.io/badge/tensorflow-keras-orange" />
      <a href="https://github.com/lycaoduong/ohlabs_pam_segmentation_unet/stargazers"><img alt="Stargazers" src="https://img.shields.io/github/stars/lycaoduong/ohlabs_pam_segmentation_unet?style=for-the-badge&logo=github&color=f4dbd6&logoColor=D9E0EE&labelColor=302D41"></a>

</p>

<p align="center">
  <img width="70%" src="figures/main.jpg">
</p>

# **Libraries Requirement**
This project depends on the following libraries:
- Tensorflow 2.2.0
- Keras 2.4.3
- Opencv 4.5.1
- Numpy 1.20.1
- Matplotlib 3.3.4
- Itk 5.1.2

# **Training**
- Run train.py 
- You can change the model on the line 80th: model = . In this project I used three kind of model:
  - U-Net
  - SegNet-5 (VGG16 backbone)
  - FCN-8 (VGG16 backbone)
  
# **Predict B-scan Image**
- Can predict all of B-scan in folder or each B-scan depends on the functions:
  - predict_all_Bscan(): predict all images in folder
  - predict_from_img(): predict desired image
  
# **Image Reconstruction**
- In volumetric_help_function.py includes 3 functions: 
  - img_2_npy(): Combine all B-scan image to 3D numpy file
  - npy_2_nrrd(): Convert numpy to NRRD format
  - cscan_reconstruct(): Reconstruct C-scan (Maximum amplitude image MAP) on 3D data
  
# **Result**
Input image | Predicted Output | 3D rendering by leveraging union of B-scan 
--- | --- | --- 
![plot](./data/test/hand/image/bscan_310.png) | ![plot](./data/test/hand/predict_colormap/bscan_310.png) | ![plot](./data/result/hand.PNG)

# **Acknowledgments**
- Any ideas on updating or misunderstanding, please send me an email: lycaoduong@gmail.com
- If you find this repo helpful, kindly give me a start!
# **Citation**
```@article{ly2022full,
  title={Full-view in vivo skin and blood vessels profile segmentation in photoacoustic imaging based on deep learning},
  author={Ly, Cao Duong and Vo, Tan Hung and Mondal, Sudip and Park, Sumin and Choi, Jaeyeop and Vu, Thi Thu Ha and Kim, Chang-Seok and Oh, Junghwan and others},
  journal={Photoacoustics},
  volume={25},
  pages={100310},
  year={2022},
  publisher={Elsevier}
