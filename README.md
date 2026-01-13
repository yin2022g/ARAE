# ARAE
We propose the Adaptive ReAdjust Edge-weight (ARAE) framework, a graph neural network method for spatial domain recognition. By incorporating an adaptive edge adjustment module, the similarity between embedding representations is used to distinguish propagated neighbor information, thereby redistributing edge weights to obtain more refined representations. Additionally, we introduce a deconvolution module for low-resolution data to enhance the model's accuracy in region recognition. Finally, we integrate a graph convolutional network that incorporates comprehensive neighborhood information. 

![Figure1](https://github.com/user-attachments/assets/50df60d0-0def-42df-a881-ed3b4bcbf42a)

## Data
### Public datasets
(1) Human DLPFCs within the spatialLIBD (http://spatial.libd.org/spatialLIBD); 
(2) Mouse anterior brain 10x Visium (https://support.10xgenomics.com/spatial-gene-expression/datasets/1.1.0/V1_Mouse_Brain_Sagittal_Anterior); 
(3) Human breast cancer(https://www.10xgenomics.com/cn/resources/datasets/human-breast-cancer-block-a-section-1-1-standard-1-1-0); 
(4) Slide-seq mouse olfactory data(https://portals.broadinstitute.org/single_cell/study/slide-seq-study); 
(5) Stereo-seq mouse olfactory data (https://github.com/JinmiaoChenLab/); 
(6) Mouse posterior brain data(https://support.10xgenomics.com/spatial-gene-expression/datasets/1.0.0/V1_Mouse_Brain_Sagittal_Posterior).
### One in-house dataset
