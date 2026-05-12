# Labratory_Work5_Comparative_Analysis_of_Pre-trained_CNN_Models_for_Custom_Image_Classification

GoogleDrive Folders: [https://drive.google.com/drive/folders/1sNmtISxANlTSLRX_j1mob1to7ax9Sv50?usp=sharing](https://drive.google.com/drive/folders/1sNmtISxANlTSLRX_j1mob1to7ax9Sv50?usp=drive_link)

Google Colab: https://colab.research.google.com/drive/14zMU2YEWuIfdS8GAFI5ADa4jCsb_jR31?usp=sharing

<img width="1344" height="584" alt="image" src="https://github.com/user-attachments/assets/39ef9b06-b3a5-4de5-9f4e-b441d9be5fd7" />

A. Model Performance
1. Which pre-trained model achieved the highest accuracy? Why?
   
EfficientNetB0 and ResNet50 both achieved the highest validation accuracy of 100% with perfect scores across all metrics. ResNet50 is considered the best overall due to its lowest validation loss of 0.0007. This is because ResNet50 uses residual connections that allow gradients to flow directly through its 50 layers, enabling it to learn very deep and complex flower features without vanishing gradient problems. Its deep architecture captures highly detailed visual patterns that generalize perfectly to unseen images.

2. Which model had the lowest performance? What could be the reason?
   
MobileNetV2 had the lowest performance with a training accuracy of 94.26% and the highest validation loss of 0.1758 compared to the other two models. This is because MobileNetV2 is a lightweight model designed for mobile devices using depthwise separable convolutions which reduce computational cost but also slightly reduce the model's capacity to learn complex features compared to EfficientNetB0 and ResNet50. Despite this it still achieved 99.90% validation accuracy which is excellent.

3. How did loss values compare across models?
   
ResNet50 achieved the lowest training loss of 0.0143 and validation loss of 0.0007, followed by EfficientNetB0 with training loss 0.0223 and validation loss 0.0016, and MobileNetV2 with the highest training loss of 0.3776 and validation loss of 0.1758. All three showed consistently decreasing loss across all 10 epochs indicating stable and healthy training with no signs of divergence.

B. Evaluation Metrics
4. Why is accuracy not enough to evaluate a model?

Accuracy alone does not show how the model performs per individual class. A model could achieve high overall accuracy by correctly predicting majority classes while completely failing on minority classes. Precision, Recall, F1-score, AUC, and Confusion Matrix provide a more complete picture by revealing whether the model is balanced across all 20 flower classes or just predicting the easier classes correctly. For example a model could get 95% accuracy while completely failing on 5 rare classes.

5. Which model had the best F1-score? What does it indicate?
   
EfficientNetB0 and ResNet50 both achieved a perfect F1-score of 1.0000 while MobileNetV2 achieved 0.9990. A perfect F1-score of 1.0000 indicates perfect balance between Precision and Recall — meaning these models correctly identified every single flower class without any false positives or false negatives across all 20 classes. This confirms that the models are not just accurate overall but equally strong on every individual class.

6. How did Precision and Recall differ across models?
   
EfficientNetB0 and ResNet50 achieved perfect Precision and Recall of 1.0000 across all 20 classes meaning zero misclassifications. MobileNetV2 achieved 0.9990 for both metrics with minor differences — Artichoke_Flower had precision of 0.98 and Protea had recall of 0.98, indicating a very small number of these flowers were slightly misclassified. Overall the differences between models were minimal showing all three are highly effective for flower classification.

C. Confusion Matrix Analysis
7. Which classes were frequently misclassified?

Based on the classification reports, only MobileNetV2 had minor misclassifications — Artichoke_Flower with precision of 0.98 and Protea with recall of 0.98 meaning a very small number of these flowers were misclassified. EfficientNetB0 and ResNet50 had absolutely zero misclassifications across all 20 flower classes showing perfect classification performance.

8. What patterns did you observe in the confusion matrix?
   
EfficientNetB0 and ResNet50 showed a perfect diagonal pattern in their confusion matrices meaning every single image was correctly classified with zero off-diagonal values. MobileNetV2 showed a near-perfect diagonal with only 1-2 misclassifications in Artichoke_Flower and Protea classes suggesting these two classes have slightly similar visual features that occasionally confuse the lighter MobileNetV2 architecture.

D. ROC and AUC
9. Which model had the highest AUC score?

EfficientNetB0 and ResNet50 both achieved a perfect AUC score of 1.0000 while MobileNetV2 also achieved 1.0000. All three models achieved perfect AUC scores meaning they perfectly distinguish between all 20 flower classes with absolutely no overlap between class probability distributions.

10. What does AUC tell us about model performance?
    
AUC measures how well a model separates different classes from each other regardless of the classification threshold. A score of 1.0000 means the model perfectly separates all 20 flower classes — it never confuses one flower for another when considering probability scores across all possible thresholds. A score of 0.5 would mean random guessing. All three models scoring 1.0000 proves excellent classification ability that goes beyond just accuracy — confirming the models are highly confident and correct in their predictions.

E. Explainability (Grad-CAM)
11. What did Grad-CAM reveal about model decision-making?

Grad-CAM revealed that all three models focused on the distinctive visual features of each flower when making classification decisions. The heatmaps highlighted petal shapes, color patterns, flower centers, and distinctive textures rather than background elements like sky or leaves. This confirms the models learned meaningful flower-specific visual features through transfer learning from ImageNet rather than relying on background patterns.

12. Did the model focus on relevant image regions?
    
Yes — all three models focused on relevant flower regions such as petals, stamens, and distinctive color patterns as shown by the Grad-CAM overlays. This confirms the models are making decisions based on actual flower features rather than background noise or irrelevant parts of the image, which is the ideal and expected behavior for a reliable flower classification system.

13. Which model produced the most meaningful heatmaps?
ResNet50 produced the most focused and precise heatmaps due to its deep 50-layer architecture with residual connections. Its heatmaps highlighted very specific and concentrated flower features with high precision. EfficientNetB0 also produced clear and well-focused heatmaps while MobileNetV2 produced slightly more diffuse heatmaps due to its lighter depthwise separable convolution architecture which captures broader features rather than highly specific ones.

F. Model Comparison and Improvement
14. Which model would you recommend for deployment? Why?

For server-based or cloud deployment I recommend ResNet50 because it achieved the lowest validation loss of 0.0007 and perfect scores across all evaluation metrics making it the most reliable and confident model. For mobile or edge device deployment I recommend MobileNetV2 because it is the most lightweight and fastest model — it requires the least memory and computing power while still achieving 99.90% validation accuracy and 1.0000 AUC, making it ideal for real-time mobile applications with limited computing resources.

15. How can you further improve your best-performing model?
    
The models already achieved near-perfect results but can be further improved by unfreezing the top layers of the base model for fine-tuning on flower-specific features, collecting more diverse and challenging training images per class to improve robustness in real-world conditions, applying stronger data augmentation like random brightness and contrast changes, using ensemble methods that combine predictions from all three models for more reliable final predictions, and converting to TensorFlow Lite or ONNX format for optimized and faster deployment.

G. Real-World Application
16. How can your model be applied in real-world scenarios?

The flower classification model can be applied in botanical gardens for automatic plant identification kiosks where visitors photograph flowers for instant identification, mobile apps for nature enthusiasts and hikers to identify wild flowers in real time, agricultural monitoring systems to track crop flower health and detect abnormalities, educational platforms for biology and botany students learning plant species, florist and e-commerce platforms for automatic flower tagging and inventory management, and environmental conservation systems to monitor rare or endangered flower species in the wild.

17. What are the risks of deploying an inaccurate model?
    
Deploying an inaccurate model could lead to misidentification of toxic or poisonous plants as safe edible species which could cause serious harm to people, wrong agricultural treatments applied to crops based on incorrect flower identification causing financial losses, loss of user trust and credibility if the app gives incorrect flower identifications, misidentification of rare or legally protected species in conservation efforts leading to environmental damage, and in pharmaceutical or herbal medicine applications misidentifying medicinal plants could lead to dangerous health consequences.

18. How can this system be integrated into a mobile/web app?
The model can be integrated by converting ResNet50 to TensorFlow Lite format for Android and iOS mobile apps enabling real-time camera-based flower identification with fast inference on device, deploying EfficientNetB0 as a REST API using Flask or FastAPI for web applications where users upload flower photos and receive instant classification results, using TensorFlow.js to run MobileNetV2 directly in the browser without a backend server for lightweight web deployment, hosting on Google Cloud AI Platform or AWS SageMaker for scalable cloud deployment that handles thousands of requests simultaneously, and building a React Native cross-platform app that captures photos sends them to the model API and returns classification results with confidence scores and flower information in real time.
