# Face Detection and Localization

This repository contains code for face detection and localization using TensorFlow and the WIDER Face dataset. The code demonstrates the process of converting XML annotations to JSON format, splitting the dataset into train, test, and validation sets, applying data augmentation, building a deep learning model using the Functional API, and making predictions on test images and real-time video.

## Code Structure

The main components of the code are as follows:

1. Data Preparation:
   - `xml_to_json`: Converts XML annotation files to JSON format.
   - `convert_annotations`: Converts annotations from the WIDER Face dataset to the desired format.
   - Dataset splitting: Splits the dataset into train, test, and validation sets.

2. Data Augmentation:
   - Uses the Albumentations library to apply various data augmentation techniques, such as random cropping, flipping, brightness and contrast adjustments, and more.
   - Augments both the images and the corresponding bounding box annotations.

3. Data Loading:
   - Defines functions to load images and labels from the augmented dataset.
   - Creates TensorFlow datasets for train, test, and validation sets.

4. Model Building:
   - Builds a deep learning model using the Functional API of TensorFlow's Keras.
   - Utilizes the VGG16 architecture as the base model for feature extraction.
   - Defines separate heads for classification and bounding box regression.

5. Training and Evaluation:
   - Compiles the model with appropriate loss functions and metrics.
   - Trains the model using the training dataset and validates on the validation dataset.
   - Plots the training and validation loss and accuracy curves.

6. Prediction:
   - Makes predictions on the test dataset and visualizes the detected faces with bounding boxes.
   - Saves the trained model for future use.

7. Real-time Detection:
   - Demonstrates real-time face detection using the trained model and OpenCV.
   - Captures video from the webcam and applies the face detection model to each frame.
   - Displays the detected faces with bounding boxes and labels in real-time.

## Dataset

The code assumes the use of the WIDER Face dataset for training and evaluation. The dataset should be organized in the following structure:
data_image/
    annotation/
    image/
    train/
        images/
        labels/
    test/
        images/
        labels/
    val/
        images/
        labels/

## Usage

1. Prepare the WIDER Face dataset and organize it according to the required structure.
2. Run the data preparation and augmentation scripts to convert annotations and augment the dataset.
3. Execute the model building, training, and evaluation code to train the face detection model.
4. Use the trained model for making predictions on test images or real-time video streams.

Please refer to the code files for detailed implementation and usage instructions.

## Dependencies

The code relies on the following dependencies:
- TensorFlow
- NumPy
- Matplotlib
- OpenCV
- Albumentations

Make sure to install the required dependencies before running the code.

## License

This code is provided under the [MIT License](LICENSE).
