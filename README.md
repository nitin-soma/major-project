# Repository For the Paper "Interpreting CNNs using Conditional GANs by representing CNNs as Conditional Priors" (Modified for Contrastive Learning)

The original paper introduced a method to interpret why a CNN makes its predictions by training a GAN to understand CNNs using GradCAM maps as supervision. This modified version removes the dependency on interpretation maps (GradCAM) and replaces external supervision with contrastive learning. The generator now learns to produce explanations directly from CNN features and input data, guided by the discriminator and a contrastive objective that encourages similar explanations for similar classes.

Key modifications:
- No GradCAM generation needed.
- Contrastive loss replaces external supervision.
- Generator learns to produce explanations directly from CNN features and data.
- Discriminator and contrastive objective guide the generator.
## Our Main Contributions via this work are :
* Introduced a GAN that understands the general working of CNNs.
* Introduced a method to represent CNN's operations as conditional priors.
* Introduced a method to interpret our proposed GAN.

## Generic Architecture
<img src = "assets/model_architecture_cropped.png">
The Generic Architecture of the proposed GAN. It takes CNN representations and Input Image to produce interpretation of CNN's predictions for the input image.

## LSFT-GAN and GSFT-GAN:
LSFT-GAN and GSFT_GAN are the two varients of the Proposed GAN architecture. They differe by thier conditioning methodologies. GSFT(Global-SFT) conditions the GAN using a Global Condition and LSFT(Local-SFT) conditions the GAN progressively.

## Results

### Sample Results
<img src = "assets/sample_figure.png" width="250" height="200">
The Figure shows sample results on interpreting CNNs trained fro Classifying Food-11 and Animals-10 Dataset.

### Interpreting Our Proposed LSFT-GAN
<img src = "assets/Individual_interpretations.png">
We interpreted how our GAN interpreted CNNs based on the relevance of the Input Conditions.

## Inference
Here me mention how to use this repository for Infererence. We provide a pretrained gan model trained on classifiction models to explain Animals-10, Food-11 and CIFAR-10 Datasets. We provide the pretrained model for both LSFT-GAN and GSFT-GAN. We provide a preprocessed .npy file of Food-11 dataset to use for Inference. Please follow the below steps for  Inference:
### Step-1 
Download the npy and pre-trained models from <a href = "https://drive.google.com/drive/folders/1PxKBHLr64gBrLFCi9N_hVNGwZUiYll8f?usp=sharing">Download Drive </a>

### Step-2
Unzip the <b>Models.zip</b> downloaded from drive to models/. <b>Models.zip</b> contains the pretained models for LSFT-GAN and GSFT-GAN
### Step-3
Move the <b>Food_Resnet.npy</b> downloaded from drive to npys/. <b>Food_Resnet.npy</b> is the npy containing the required preprocessed inputs for Inference. 
### Step-4
Create a new python environment and run requirements.txt
In command line:
~~~
py venv -m env
env/scripts/activate.bat
pip install -r requirements.txt
~~~
### Step -5
To Run Inference on LSFT-GAN run
~~~
python lsft_inference.py
~~~

To Run Inference on GSFT-GAN run
~~~
python gsft_inference.py
~~~
### Step -6
Open outputs/ to find the folder with the inferences.
The inference folder contains three sub floders inputs , Gan outputs and GradCAM outputs.
