# Parking Lot Monitor

This project focuses on improving urban parking management by developing a machine-learning-based system to identify real-time occupancy status of parking spaces.

## Network Architecture

The model uses a CNN architecture which processes images to classify parking space occupancy. It includes 3 convolution layers and 3 pooling layers to extract and learn features from the input images, with dropout layers to prevent overfitting.

<img src="D:\UCL\CASA0018\CASA0018-Parking-Space-Detection\Report\diagrams\architecture.png" />



## Data

The primary dataset used is the PKLot dataset from ufpr, which includes over 12,000 images of parking lots annotated with occupancy status under different weather conditions. 

This is a sample:

<img src="D:\UCL\CASA0018\CASA0018-Parking-Space-Detection\Report\diagrams\data_sample.png" />

You can find full dataset here: [Parking Lot Database – Laboratório Visão Robótica e Imagem (ufpr.br)](https://web.inf.ufpr.br/vri/databases/parking-lot-database/)



### System Workflow

1. **Data Collection**: Utilizes the PKLot dataset with images under various weather conditions for training the model.
2. **Model Development**: Trains a CNN to differentiate between occupied and unoccupied parking spaces. The model adjusts parameters through iterative training.
3. **Deployment**: Uses a Raspberry Pi 4B with an external camera to monitor parking lots and update occupancy status via an MQTT server.

<img src="D:\UCL\CASA0018\CASA0018-Parking-Space-Detection\Report\diagrams\application_diagram.png" />

<img src="D:\UCL\CASA0018\CASA0018-Parking-Space-Detection\Report\diagrams\workflow.png" />

## Closure

<img src="D:\UCL\CASA0018\CASA0018-Parking-Space-Detection\Report\diagrams\raspi1.jpg" />

<img src="D:\UCL\CASA0018\CASA0018-Parking-Space-Detection\Report\diagrams\raspi2.jpg" alt="raspi2" />

This is a really really good Raspberry Pi 4B case with camera mount, you can get it here:
[Raspberry Pi 4 Camera Case by bkahan - Thingiverse](https://www.thingiverse.com/thing:4555651)

The models are also contained in the project folder.



## Quick Start

Firstly, clone this repo from github:
```
git clone https://github.com/YikunLi9/CASA0018-Parking-Space-Detection.git
```

If you want  to test the model I trained, please:
```
cd ./Projects/Final Project
python3 ./model_test.py
```

Or, if you want to train your own model, please prepare your dataset and divide them into training and validation set in folder with name 1 and 0 and try my notebook, I prefer Jupyter Notebook, or you can also use colab or other tools:

```
cd ./Projects/Final Project
Jupyter-Notebook
```



## Contact

If you have any question about the project or want to contribute to it, welcome to contact with me:

yikun.li.22@ucl.ac.uk
