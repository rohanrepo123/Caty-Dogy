# Cats vs Dogs Image Classification with CNN and VGG16

An image-classification project that compares custom convolutional neural networks with VGG16 transfer learning, image augmentation, and fine-tuning. The models are trained in Google Colab using TensorFlow/Keras and the Kaggle Cats and Dogs dataset.

## Project Overview

The notebook builds a binary classifier that predicts whether an input image contains a cat or a dog. It starts with CNN models trained from scratch, then improves generalization using a pretrained VGG16 feature extractor and fine-tuning.

Class labels:

- `0` — Cat
- `1` — Dog

## Dataset

The project uses the [Cat and Dog dataset by tongpython on Kaggle](https://www.kaggle.com/datasets/tongpython/cat-and-dog).

| Split | Images | Classes |
|---|---:|---:|
| Training | 8,005 | 2 |
| Test/validation | 2,023 | 2 |

Images are resized to `256 × 256`, loaded in batches of 32, and normalized for the custom CNN models.

## Approaches Explored

### 1. Custom CNN

- Three convolutional blocks with ReLU activation
- Batch normalization and max pooling
- Two dense layers with dropout
- Sigmoid output for binary classification
- Adam optimizer and binary cross-entropy loss

The first CNN reaches very high training accuracy but substantially lower validation accuracy, demonstrating overfitting.

### 2. Improved CNN with Early Stopping

A second CNN changes the filter progression to `32 → 64 → 128` and uses early stopping on validation accuracy. The callback restores the weights from the best epoch.

### 3. VGG16 Transfer Learning

VGG16 pretrained on ImageNet is used without its original classification head. Its convolutional base is frozen and connected to new dense layers for cats-vs-dogs classification.

### 4. Image Augmentation

The notebook generates additional training samples using small rotations, shear, zoom, horizontal flips, and width/height shifts. This stage is evaluated with the frozen VGG16 model.

### 5. VGG16 Fine-Tuning

Layers from `block5_conv1` onward are unfrozen so the final VGG16 block can adapt to the dataset. Early stopping restores the best-performing weights.

## Results

| Model | Best recorded validation accuracy |
|---|---:|
| Custom CNN | ~73.6% near the end of training |
| Improved CNN | Best weights restored from epoch 14 |
| Frozen VGG16 | 92.88% |
| VGG16 with augmentation | 92.68% |
| Fine-tuned VGG16 | **95.06%** |

The fine-tuned model produced the strongest recorded validation performance. Its lowest recorded validation loss was `0.1246` at epoch 4, whose weights were restored by early stopping.

## Technologies

- Python
- TensorFlow and Keras
- VGG16 / ImageNet transfer learning
- OpenCV
- NumPy and pandas
- Matplotlib
- KaggleHub and OpenDatasets
- Google Colab with GPU acceleration

## Running the Notebook

1. Open `CNN_(4).ipynb` in Google Colab.
2. Enable a GPU runtime from **Runtime → Change runtime type**.
3. Run the dataset-download cell and provide Kaggle credentials if prompted.
4. Confirm that the dataset is available under `/content/cat-and-dog/`.
5. Run the remaining cells in order to train and compare the models.
6. Upload `.jpg` images to `/content` to test cat/dog predictions.

The main packages can be installed with:

```bash
pip install tensorflow opencv-python numpy pandas matplotlib kagglehub opendatasets
```

## Prediction Logic

The binary sigmoid output is interpreted using a threshold of `0.5`:

```python
prediction = model.predict(image)
label = "dog" if prediction[0][0] > 0.5 else "cat"
```

## Notebook Notes

- The notebook uses Colab-specific paths and shell commands.
- VGG16 inputs should consistently use `tensorflow.keras.applications.vgg16.preprocess_input`.
- The test set is used as validation data during training, so the reported figures should be treated as validation results rather than an independent final test score.
- The notebook contains saved image outputs, which makes the `.ipynb` file relatively large.

## Future Improvements

- Create separate training, validation, and test splits.
- Add precision, recall, F1-score, and a confusion matrix.
- Use `GlobalAveragePooling2D` to reduce parameters and overfitting.
- Save the best model and provide a reusable inference script or web interface.
- Apply augmentation dynamically in the training pipeline instead of writing generated files into the dataset directory.

## Acknowledgements

- Dataset: [Kaggle Cats and Dogs](https://www.kaggle.com/datasets/tongpython/cat-and-dog)
- Pretrained architecture: VGG16 from TensorFlow/Keras Applications
