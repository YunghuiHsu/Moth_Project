# Moth_Project
For Moth Project backup in  BRCAS Laboratory for Ecology and Sociobiology Studies(ESS) Lab


- Using fine-tuned object detection model (YOLOv4) for Lepidoptera to frame the exact location of specimen images

- Obtain [background removal images of Lepidoptera specimens](https://hackmd.io/@YungHuiHsu/Hkr3R3wV9) (`Moth_thermal/remove_bg/`)
  - Obtain a small number of rough images of the background removal images using "Unsupervised Image Segmentation by Backpropagation" model
  - Supervised training of the model (UNet) from scratch, incremental increase, and accurate background removal images
- [Variational Sparse Coding (VSC)](https://hackmd.io/@YungHuiHsu/HJN5IL2gs) in Latent space based on variational autoencoders (VAEs) is used to improve feature interpretability for subsequent analysis of deep features(`Moth_thermal/vsc/`)
