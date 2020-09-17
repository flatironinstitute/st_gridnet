# GridNet: A convolutional neural network for common coordinate registration of high-resolution histopathology images

## Introduction
This is an implementation of the model described in our publication "A convolutional neural network for common-coordinate registration of high-resolution histology images," which was developed principally for applications to registration of data collected during solid-phase capture spatial transcriptomics (ST) experiments.

## Prerequisites
* Python (3.7)
* PyTorch (1.4.0)
* torchvision (0.5.0)
* NumPy (1.16.4)
* Pillow (6.0.0)
* pandas (0.24.2)
* sklearn (0.22.2)
* matplotlib (3.1.0)
* HexagDLy (https://github.com/ai4iacts/hexagdly)

## Running the code

### Generating training data

For each tissue of interest, GridNet expects the following information:
* Whole-slide image file
* Locations of foreground (tissue-containing) spots in ST grid
* Class assignments for each foreground spot (if training a new model)

At a low level, training data are expected to be provided in two directories with the following structure:
```
data
|___imgs
|   |___tissue1
|   |   |   1_2.jpg
|   |   |   1_4.jpg
|   |   |   ...
|   |  
|   |___tissue2
|   |   |   1_2.jpg
|   |   |   2_2.jpg
|   |   |   ...
|   |
|   |___...
|  
|___lbls
    |   tissue1.png
    |   tissue2.png
    |   ...
```
For each tissue in the training set, there should exist a sub-directory within the ```imgs``` directory that contains extracted patch images, and an identically-named PNG file in the ```lbls``` directory that contains label information. The label image should have dimensions equal to the dimensions of the ST array (33x35 for standard ST, 64x78 for Visium). All pixels in the label image should contain an integer value between 0 and N_class, with 0 pixels indicating that the corresponding spot belongs to the slide background and nonzero pixels indicating the class label for the corresponding spot. For each foreground spot location (xind,yind), there should exist a correpsonding patch image file within the relevant sub-directory of ```imgs``` named according to ```xind_yind.jpg```.

In order to simplify the processing of inputs to GridNet, we have provided modules that generate these directories from either Cartesian ST or Visium ST output files.

#### Cartesian ST

For data obtained from Cartesian ST experiments, we have provided a module that interfaces with annotations files of the format employed by Maniatis et al. [[1]](#1). For each tissue image, a corresponding tab-separated value (TSV) file of the following format:

|        | x0_y0 | x1_y1 | ...  | xN_yN |
| ------ | ----- | ----- | ---- | ------|
| Class1 | 0     | 1     | ...  | 0     |
| Class2 | 1     | 0     | ...  | 1     |
| ...    | ...   | ...   | ...  | ...   |
| ClassK | 0     | 0     | ...  | 0     |

where column names indicate the indices of foreground (tissue-containing) spots in the ST array, and the column vectors are one-hot encoded to indicate the class membership of each spot.

#### Visium ST

For data obtained from Visium ST experiments, we have provided a module that interfaces directly with the outputs from 10x Genomics' [SpaceRanger](https://support.10xgenomics.com/spatial-gene-expression/software/pipelines/latest/output/images) and [Loupe](https://support.10xgenomics.com/single-cell-gene-expression/software/visualization/latest/tutorial-interoperability).

### Training a model

We provide a Jupyter notebook, ```gridnet_training_example.ipynb```, that details the process by which you may go about training a GridNet model. Additionally, all scripts used to generate the models and figures in our publication are included in ```publication/scripts```.

### Predicting on new data

## Citation

If you found this repository useful, please cite us!

## References

<a id="1">[1]</a> 
Maniatis, S., Aijo, T., Vickovic, S., Braine, C., Kang, K., Mollbrink, A., Fagegaltier, D., Andrusivova, Z., Saarenpaa, S., Saiz-Castro, G., Cuevas, M., Watters, A., Lundeberg, J., Bonneau, R., Phatnani, H. (2019). 
Spatiotemporal dynamics of molecular pathology in amyotrophic lateral sclerosis. 
Science, 364(6435), 89-93.
doi:10.1126/science.aav9776
