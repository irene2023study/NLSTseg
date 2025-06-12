# Lung Cancer Segmentation Model Using U-Net

This repository contains a medical image prediction model that performs **lung cancer segmentation** from CT scans using a **U-Net architecture**. The model was trained on the **NLSTseg** dataset, which is derived from the **NLST** (National Lung Screening Trial) dataset. This work aims to provide a tool for detecting lung cancer at a pixel-level segmentation.

## Overview

The model utilizes a **U-Net** neural network to segment lung cancer lesions in **CT images**. The **NLSTseg** dataset, created by the author, contains CT scans of patients diagnosed with lung cancer. The dataset consists of 715 lung lesions manually annotated with pixel-level segmentation labels to highlight the affected areas.

While the model demonstrates impressive performance on the training set (IOU = 0.95), the performance on the test set is lower (IOU = 0.42). Despite the lower performance on the test set, the model is made publicly available to assist future research in this area and to serve as a reference for others looking to tackle similar challenges.

## Dataset: NLSTseg

The **NLSTseg** dataset was created by extracting **CT images** from the **NLST (National Lung Screening Trial)**, which is publicly available. For this project, only the images diagnosed as **lung cancer** were selected, and the lung cancer lesions were manually annotated at the pixel level.

- **Total annotated lung lesions:** 715
- **Input data:** CT images of lung cancer patients from the NLST dataset
- **Segmentation:** Pixel-level annotations marking the lung cancer lesions

## Model Architecture

The model is based on the **U-Net** architecture, which is widely used for image segmentation tasks, especially in medical imaging. The U-Net consists of an encoder-decoder structure that captures both high-level and low-level features to make precise predictions.

### Key Components:
- **Encoder:** Downsampling the input image to extract features.
- **Bottleneck:** The central layer of the network that connects the encoder and decoder.
- **Decoder:** Upsampling the feature maps to generate the segmentation mask.
- **Skip Connections:** The skip connections between the encoder and decoder help preserve spatial information.

## Results

- **Training Set Performance:**
  - **IOU:** 0.95 (Excellent performance)
  
- **Test Set Performance:**
  - **IOU:** 0.42 (Model shows lower generalization performance on the test set)

While the model achieves high performance on the training set, its performance on the test set indicates there may still be room for improvement. Further model tuning, data augmentation, or other techniques may help increase its generalization ability.

## Usage

To use the trained model for lung cancer segmentation, follow these steps:

1. Clone this repository:
   ```bash
   git clone https://github.com/irene2023study/NLSTseg/tree/main.git
   cd Unet_seg
