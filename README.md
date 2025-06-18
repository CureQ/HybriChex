# HybriChex

# Overview
HybriChex is a research project developed as part of the Applied Artificial Intelligence (AI) master thesis at the Amsterdam University of Applied Sciences (HvA). The project explores a hybrid approaches, focusing on the combination of a CNN + Swin Transformer for chest X-rays trained on the NIH Chest-Xray14 dataset from the NIH.

# Project Structure
The repository is organized as following:
- GUI: This directorty contains the model integrated into a Grapical User Interface;
- Model Iterations: The performed iterations on the development of the hybrid CNN Swin Transformer model;
- data: The official train-test split from the orignal dataset;
- Weights_chexnet: A link to the weights for the ChexNet CNN;
- Weight_SwinChex: A link to the weights for the SwinChex Vision Transformer;
- Finalmodel.ipynb: The final model after the iterations;
- Dataload.ipynb: A notebook to load the NIH Chest-Xray14 dataset;
- Environment.yaml: A environment file containing the necessary libaries to run the finalmodel.ipynb;
- Exploratory_data_analysis.ipynb: A file containing a EDA on the dataset.

# Setup
To set up the project locally:
- Clone the repository --> git clone https://gitlab.fdmci.hva.nl/steenhr1/aai_thesis_hybrichex.git
- Install dependecies --> For model: environment.yaml For GUI: pip install -r requirements.txt

# Disclaimer AI usage
This work has benefited from the use of AI language models for initial checks on spelling, grammar, and layout. However, all content has been thoroughly reviewed, rewritten, and validated in my own words. I retain full responsibility for the accuracy, originality, and overall quality of this work.

