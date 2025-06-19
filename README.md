# Smile Classification

This project implements a smile classification model using the CelebA dataset and PyTorch. The goal is to classify whether a person in an image is smiling or not.

## Dataset

- **CelebA Dataset**: A large-scale face attributes dataset with more than 200,000 celebrity images, each with 40 attribute annotations. This project uses the 'Smiling' attribute for binary classification.
- Download the dataset from the [official CelebA website](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) or use a script to automate the download.

## Model Architecture

- The input images are passed through four convolutional layers:
  - **Conv1**: 32 feature maps, 3×3 kernel, padding=1
  - **Conv2**: 64 feature maps, 3×3 kernel, padding=1
  - **Conv3**: 128 feature maps, 3×3 kernel, padding=1
  - **Conv4**: 256 feature maps, 3×3 kernel, padding=1
- The first three convolutional layers are each followed by 2×2 max pooling.
- Two dropout layers are included for regularization.
- The final output is a binary classification (smile / no smile).

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- numpy
- matplotlib (for visualization)

Install dependencies with:
```bash
pip install torch torchvision numpy matplotlib
```

## Usage

1. **Download the CelebA dataset** and place it in the appropriate directory.
2. **Run the notebook**:
   - Open `smile_classification.ipynb` in Jupyter Notebook or JupyterLab.
   - Follow the cells to preprocess data, train the model, and evaluate performance.

## Project Structure

- `smile_classification.ipynb`: Main notebook containing all code for data loading, preprocessing, model training, and evaluation.

## Results

- The model achieves smile classification by learning from facial features in the CelebA dataset.
- For detailed results and visualizations, see the output cells in the notebook.

## Acknowledgements

- [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- [PyTorch](https://pytorch.org/)

---
