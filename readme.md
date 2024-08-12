# Emotion Detection

This application predicts your emotions using live feed from your local computer webcam. For this app, I have developed several Convolutional Neural Networks (CNNs) from scratch and selected the one with the highest accuracy.

# About Data

We have a total of 35,887 (48x48) grayscale images across 7 classes.

![Data Analysis](artifacts\data.png)



## Models Used

1. **Custom CNN from Scratch Without Data Augmentation**
2. **Custom CNN from Scratch With Data Augmentation**
3. **VGG16**
4. **ResNet50**

The metrics for the best-performing CNN are as follows:

1. **Accuracy Plot** 

   ![Accuracy Plot](artifacts\acc.png)

2. **Loss Plot**

   ![Loss Plot](artifacts\loss.png)

3. **Confusion Matrix**

   ![Confusion Matrix](artifacts\cm.png)




## How to Use This Application

1. Clone the repository.
2. Run the command:
   ```bash
   python emotion_detection.py