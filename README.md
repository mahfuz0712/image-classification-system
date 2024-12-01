
# Image Classification Project

This project implements an image classification system using TensorFlow and Keras. It trains a Convolutional Neural Network (CNN) on a custom dataset and saves the best model as `best.h5`.

## Folder Structure

```
Image Classification Project/
├── main.py
├── src/
│   ├── data_preprocessing.py
│   ├── train.py
│   └── evaluate.py
├── models/  # Trained model will be saved here
├── datasets/
│   ├── train/  # Training data
│   └── test/   # Testing data
```

## Requirements

Install the required Python packages:

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

## How to Run

1. Ensure your dataset is organized as follows:
   ```
   datasets/
   ├── train/
   │   ├── class1/
   │   ├── class2/
   └── test/
       ├── class1/
       ├── class2/
   ```

2. Train and evaluate the model:
   ```bash
   python main.py
   ```

3. After training, the best model will be saved in the `models` folder as `best.h5`.

## Evaluation

The evaluation script calculates the **AUC Score** for the test dataset. The result will be displayed in the console.

## Troubleshooting

- Ensure your dataset directory matches the required structure.
- If the script cannot find the model, check the `models/` folder and the model's file name.

## License

This project is licensed under the MIT License.
