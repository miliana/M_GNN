M-GNN: Graph Neural Network for Lung Cancer Detection
Overview
This repository contains the implementation of M-GNN, a Graph Neural Network framework designed for early lung cancer detection using metabolomics and heterogeneous graph modeling, as described in the study "M-GNN: A Graph Neural Network Framework for Lung Cancer Detection Using Metabolomics and Heterogeneous Graph Modeling" (Vaida et al., 2025). The model leverages GraphSAGE and GAT layers to analyze a heterogeneous graph integrating metabolomics data from 800 plasma samples (586 cases, 214 controls), demographic features, and Human Metabolome Database (HMDB) annotations. The provided code snippet, shap_analysis.py, computes SHAP (SHapley Additive exPlanations) values to identify key predictors of lung cancer, such as cigarette pack years, choline, and taurine, achieving a test accuracy of 89% and an ROC-AUC of 0.92.
Prerequisites
To run this code, you’ll need the following dependencies:

Python 3.8+
PyTorch (torch>=1.9.0)
PyTorch Geometric (torch-geometric>=2.0.0)
SHAP (shap>=0.41.0)
NumPy (numpy>=1.21.0)

Install the dependencies using pip:
pip install torch torch-geometric shap numpy

Code Description
The file shap_analysis.py performs feature importance analysis using SHAP values, a critical step in interpreting the M-GNN model’s predictions. It uses a subset of the training data as a background for the SHAP KernelExplainer and computes SHAP values for the test set to quantify the contribution of each feature (e.g., metabolite expression levels, demographic variables) to lung cancer predictions. This aligns with the manuscript’s Results section, where SHAP identified cigarette pack years, choline, and taurine as top predictors, reflecting smoking and metabolic dysregulation in lung cancer.
Code Snippet (shap_analysis.py)
import shap

# Assuming model_predict is a function that takes inputs and returns model predictions
# and data is a PyTorch Geometric Data object with x, train_mask, test_mask
background = data.x[data.train_mask].cpu().numpy()[:100]  # Subset of training data as background
test_inputs = data.x[data.test_mask].cpu().numpy()

Usage

Prepare the Data and Model: Ensure you have a trained M-GNN model and a PyTorch Geometric Data object containing the heterogeneous graph, with node features (data.x), train mask (data.train_mask), and test mask (data.test_mask). The graph should include patient, metabolite, pathway, and disease nodes, as described in the manuscript (Materials and Methods, Figure 2).
Define the model_predict Function: Create a function that takes input features (numpy array) and returns the model’s predictions. For example:def model_predict(x):
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x, dtype=torch.float).to(device)
        return model(x_tensor).cpu().numpy()

Here, model is the trained M-GNN model, and device is the device (CPU/GPU) your model is on.



Notes

Data Availability: Due to privacy and ethical restrictions, the raw metabolomics data from the 800 plasma samples is not available (see manuscript, Data Availability Statement). However, the graph structure can be replicated using HMDB annotations and synthetic data, as described in the manuscript.
Relational Context: The M-GNN model leverages the relational context of metabolite-pathway-disease connections (Materials and Methods). This code snippet focuses on post-training analysis, but the full model implementation includes graph construction and training, which can be extended based on the manuscript’s description.

License
This project is licensed under the MIT License. See the LICENSE file for details (to be added).
Citation
If you use this code in your research, please cite:

Vaida, M.; Wu, J.; Himdiat, E.; Haince, J.-F.; Bux, R.A.; Huang, G.; Tappia, P.S.; Ramjiawan, B.; Ford, W.R. M-GNN: A Graph Neural Network Framework for Lung Cancer Detection Using Metabolomics and Heterogeneous Graph Modeling. Int. J. Mol. Sci., 2025.

