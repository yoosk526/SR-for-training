# Real-time Super Resolution in Drone video (Train)  
  
This repository is for training model.  
We trained our model in **Docker Container** *(CUDA 11.5).*  
If you do **git clone** and upload this repository to your Google Drive, you can train model on *Colab*.  
*(`SR_train_on_Colab.ipynb` is code for training on Colab, but it might raise error)*  
Generally, we recommend using Docker to train. If you have great GPU, you don't need it.

## Environment
|Name|Version|
|---|---|
|Server|A100|
|CUDA|11.5|
|python||
|torch||
|torchvision||
|||

## Dataset

We use VisDrone dataset instead of DIV2K, and it resulted in better image performance.  
`visdrone_setup.sh` & `rename_move.py` are codes for downloading and arranging the Dataset.  
(Execute `visdrone_setup.sh` **first**)  

### Augmentation methods
...

## Train 
We propose a **multi-stage warm-start** training strategy which accelerates the convergence and improves the final performance.  
In the first stage, we train ABPN from scratch.  
Then in the next stage, instead of training from scratch, load the weights of ABPN of the previous stage.  
In this second stage, we use also **QAT** *(Quantization Aware Training)*.  
- Example  

    ```
    # 1st stage
    python train.py --preload --epochs 200

    # 2nd stage
    python train.py --preload --load [pretrained weight] --epochs 300 --qat
    ```
