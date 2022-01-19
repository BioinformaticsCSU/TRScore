# TRScore

## source codes: https://github.com/BioinformaticsCSU/TRScore
## What is TRScore?
** TRScore is a scoring method for protein-protein docking, which combined 3D convolutional ResNet and physicochemical features of protein atoms.**

## Abstract 

Protein-protein interactions (PPIs) play indispensable roles in cellular activities. Due to the technical difficulty and high cost, there is considerable interest toward the development of computational approaches to decipher PPI patterns, such as protein docking. One of the most important and difficult aspects of protein docking is distinguishing near-native conformations from decoys, but unfortunately existing scoring methods suffer from poor accuracy.

TRScore is a scoring method that helps to re-rank protein-protein docking decoys, which distinguish more near-native (acceptable, medium and high in the CAPRI-criteria) decoys from non-natives.

To distinguish a near-native model from decoy, TRScore voxelizes the protein-protein interface into a 3D grid labeled by the number of atoms in different physicochemical classes. Benefit from the RepVGG based on deep convolutional ResNet architecture, TRScore can effectively extract information from physicochemically characterized grids and discriminate energetic-favoured near-native models from energetic-unfavoured non-native decoys.

TRScore was extensively evaluated on diverse test sets including the ZDOCK benchmark version 5.0 and the DockGround unbound decoy set. It was shown that TRScore significantly outperformed over existing approaches in terms of both success rate and enrichment factor. Given its magnificent performance, it is anticipated that TRScore will serve as an indispensable scoring function for protein-protein docking decoys.


## data preprocessing:
![Fig_1](https://github.com/BioinformaticsCSU/TRScore/blob/master/Fig_1.png)

## Architecture of TRScore:
![Fig_2](https://github.com/BioinformaticsCSU/TRScore/blob/master/Fig_2.png)

## Results on DockGround decoys set:
![Fig_3](https://github.com/BioinformaticsCSU/TRScore/blob/master/Fig_3.png)

(Only performance in DockGround decoys set, more details in paper please)

## Reference:
```
ZRANK, ZRANK2, IRAD refer: https://zdock.umassmed.edu/software/
DOVE: https://github.com/kiharalab/DOVE/
```

# Codes dependencies:
* CUDA==10.1
* cudnn==7.6.5
* pytorch==1.8

# File explanation:
* main.py: main program for predicting probabilities of input decoys (PDB files).
* src/AtomTypeDictionary.py: the Dictionary of atom types using in TRScore.
* src/BuildModel.py: source codes of model structure of TRScore.
* src/prepare.py: to voxelize protein-protein interface from input decoys (PDB files) into 3D grid labeled by the number of atoms in different physicochemical classes from AtomTypeDictionary.py. Additionally, these 3D grids (NPZ files) will be stored in working directory.
* src/predict.py: source codes of predicting functions used in main.py.
* examples: the folder stored some examples.
* models: the folder stored 4-fold models trained by our lab.

# Usage
```
main.py:
-F: path of decoy example (PDB files)
-M: path of model file (.pt file)
-W: path of working directory
--gpu: to Choose gpu id (default=0)
```
```
prepare.py:
-F: path of example (PDB files)
-O: path of saving NPZ files
```
## 1. predict all PDB files in examples/complexes
```
python main.py -F examples/complexes -M models/model1.pt -W examples/NPZ
```
## 2. prepare your own NPZ files
```
python prepare.py -F examples/complexes -O examples/NPZ
```
