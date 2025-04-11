# Celebrity Facial Recognition

A facial recognition system built with PyTorch and fastai that can identify celebrities from images.

## Overview

This project uses deep learning to identify celebrities from facial images. It is trained on the CelebA dataset, which contains over 200,000 celebrity images with identity labels.

## Features

- Train a deep learning model to recognize celebrities
- Analyze dataset statistics and view sample images
- Evaluate model performance with detailed metrics
- Web interface for real-time celebrity recognition from uploaded images

## Requirements

- Python 3.7+
- PyTorch 1.9+
- fastai 2.4+
- Gradio (for web interface)
- Other dependencies listed in requirements.txt

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Dataset

The project uses the CelebA dataset, which should be organized as follows:
- `img_align_celeba/`: Directory containing aligned celebrity face images
- `list_identity_celeba.txt`: File mapping image IDs to celebrity identities

## Usage

### Data Analysis

To analyze the dataset and get statistics:

```
python data_preparation.py
```

This will generate statistics about the dataset and save sample images of celebrities.

### Training

To train the facial recognition model:

```
python celebrity_recognition.py
```

The trained model will be saved to `models/celebrity_recognition_model.pkl`.

### Evaluation

To evaluate the model's performance:

```
python model_evaluation.py
```

This will generate evaluation metrics, confusion matrices, and other visualizations in the `evaluation_results` directory.

### Web Interface

To launch the web interface for celebrity recognition:

```
python web_interface.py
```

This will start a Gradio web interface where you can upload photos to identify celebrities.

## Project Structure

- `celebrity_recognition.py`: Main script for training the model
- `data_preparation.py`: Utilities for analyzing and preparing the dataset
- `model_evaluation.py`: Tools for evaluating model performance
- `web_interface.py`: Gradio web interface for using the trained model
- `models/`: Directory for storing trained models
- `evaluation_results/`: Directory for storing evaluation metrics and visualizations
- `sample_images/`: Directory containing sample images from the dataset

## Performance

The model is trained using a ResNet-50 architecture fine-tuned on the CelebA dataset. Performance metrics will vary depending on the number of celebrities in the dataset and training parameters.

## License

This project is for educational purposes only. The CelebA dataset comes with its own license that must be respected.

## Acknowledgments

- CelebA dataset creators
- fastai and PyTorch communities 