# Emotion Detection

This application predicts your emotions using live feed from your local computer webcam. For this app, I have developed several Convolutional Neural Networks (CNNs) from scratch and selected the one with the best performance.

# About Data

We have a total of 35,887 (48x48) grayscale images across 7 classes.


![data](https://github.com/user-attachments/assets/c9f0ed3d-f103-48d2-805e-e6cff1b137f7)



## Models Used

1. **Custom CNN from Scratch Without Data Augmentation**
2. **Custom CNN from Scratch With Data Augmentation**
3. **VGG16**
4. **ResNet50**

The metrics for the best-performing CNN are as follows:

1. **Accuracy Plot** 

   ![acc](https://github.com/user-attachments/assets/5fa644be-304a-4dc3-82c8-12418c72a4e3)


2. **Loss Plot**

   ![loss](https://github.com/user-attachments/assets/332df7b3-7853-4d10-81f4-37455d49cfa9)

4. **Confusion Matrix**

   ![cm](https://github.com/user-attachments/assets/7a095bcb-d9e4-47c1-9929-9811a701daec)




## How to Use This Application

1. Clone the repository.
2. Run the command:
   ```bash
   python emotion_detection.py
