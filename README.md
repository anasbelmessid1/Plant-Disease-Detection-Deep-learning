# Plant Disease Detection using Deep Learning

## Project Overview
This project aims to develop a deep learning model to detect and diagnose plant diseases from images. The model leverages Convolutional Neural Networks (CNNs) to identify various plant diseases and provides treatment recommendations. The project compares the custom architecture with pre-trained models like VGG16, EfficientNetB0, and ResNet50.

## Abstract
Plant diseases pose a significant threat to agriculture, affecting crop yield and quality. Early detection and diagnosis are crucial for effective disease management. This project utilizes deep learning techniques to create a robust model for identifying plant diseases from images. The model's performance is compared with popular pre-trained architectures to determine the most effective solution.

## Video Demonstration (Click on the picture to watch the video)
[![Watch the video](https://img.youtube.com/vi/DrzZybT_Fig/maxresdefault.jpg)](https://www.youtube.com/watch?v=DrzZybT_Fig)

## Screenshots
Here are some screenshots of the app in action:

### Home Screen
![Home Screen](./screenshots/App%20Homepage.PNG)

### Disease Detection Page 
![Disease Detection](screenshots/Disease%20Detection%20Page.PNG)

### Results
![Results](./screenshots/results.png)

## Acknowledgments
We express our gratitude to our Supervisor for their guidance and support throughout this project.

## Project Structure
1. **Introduction**
   - Importance of plant disease detection
   - Overview of traditional and modern detection methods

2. **Literature Review**
   - Traditional plant disease detection techniques
   - Advances in deep learning for plant disease detection

3. **Theoretical Basis**
   - Convolutional Neural Networks (CNNs)
   - Transfer learning and pre-trained models (VGG16, EfficientNetB0, ResNet50)

4. **Methodology**
   - Data collection and preprocessing
   - Model architecture and training
   - Comparison with pre-trained models

5. **Results and Discussion**
   - Performance metrics
   - Comparison of custom model with pre-trained models

6. **Conclusion**
   - Summary of findings
   - Future work and potential improvements

## Detailed Methodology

### Data Collection
A comprehensive dataset of plant images was collected, including both healthy and diseased samples. The images were annotated with labels indicating the type of disease.

### Data Preprocessing
- **Image Augmentation**: Techniques such as rotation, flipping, and scaling were applied to increase the dataset size and variability.
- **Normalization**: Image pixel values were normalized to improve model convergence.

### Model Architecture
- **Custom CNN Architecture**: Designed specifically for plant disease detection, featuring multiple convolutional layers, batch normalization, and dropout for regularization.
- **Pre-trained Models**: VGG16, EfficientNetB0, and ResNet50 were fine-tuned on the plant disease dataset.

### Training
- **Optimizer**: Adam optimizer was used for training.
- **Loss Function**: Categorical cross-entropy loss was employed.
- **Evaluation Metrics**: Accuracy, precision, recall, and F1-score were used to evaluate model performance.

### Comparison with Pre-trained Models
- **VGG16**: Known for its simplicity and depth, providing a strong baseline.
- **EfficientNetB0**: Balances model size and accuracy, known for efficient scaling.
- **ResNet50**: Introduces residual learning, allowing training of deeper networks.

## Results
The custom model and pre-trained models were evaluated on the test dataset. Performance metrics were compared to determine the most effective architecture for plant disease detection.

### Performance Comparison
| Model                | Model Size | Training Accuracy (%) | Validation Accuracy (%) | Test Accuracy (%) | Training Time | Inference Time per Image |
|----------------------|------------|-----------------------|-------------------------|-------------------|---------------|--------------------------|
| EfficientNetB0       | 29.6 MB    | 98.57                 | 96.93                   | 96.9              | 1h 30m        | 0.06s                    |
| VGG16                | 528 MB     | 87.28                 | 84.79                   | 84.81             | 2h 15m        | 0.07s                    |
| ResNet50             | 98 MB      | 98.26                 | 96.25                   | 96.60             | 1h 45m        | 0.08s                    |
| Custom Model         | 2.3 MB     | 97.20                 | 96.12                   | 96.8              | 2h 5m         | 0.06s                    |
| Hybrid Custom Model  | 45.1 MB    | 99.89                 | 98.32                   | 98.67             | 2h 20m        | 0.06s                    |

## Conclusion
The custom CNN architecture demonstrated competitive performance compared to pre-trained models. The project highlights the potential of deep learning for accurate and efficient plant disease detection.

### Future Work
- Further optimization of the custom model.
- Exploration of additional datasets.
- Integration of the model into a user-friendly web and mobile application for real-time disease detection.
