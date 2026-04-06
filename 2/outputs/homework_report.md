# Pattern Recognition and Machine Learning Homework 2

## Experimental Setup
- Training set: 1000 samples, balanced across C0 and C1.
- Test set: 500 samples, balanced across C0 and C1, generated from the same distribution.
- Data noise standard deviation: 0.2.
- Model selection uses 5-fold stratified cross validation on the training set.
- Compared models: Decision Tree, AdaBoost + Decision Tree, and SVM with linear, RBF, polynomial, and sigmoid kernels.

## Aggregate Results

| model_name               |   best_cv_accuracy |   test_accuracy |   test_precision |   test_recall |   test_f1 | best_params                                                            |
|:-------------------------|-------------------:|----------------:|-----------------:|--------------:|----------:|:-----------------------------------------------------------------------|
| AdaBoost + Decision Tree |              0.975 |           0.972 |         0.979675 |         0.964 |  0.971774 | {'estimator__max_depth': 2, 'learning_rate': 1.0, 'n_estimators': 200} |
| SVM (RBF kernel)         |              0.979 |           0.972 |         0.972    |         0.972 |  0.972    | {'svc__C': 10, 'svc__gamma': 'scale'}                                  |
| Decision Tree            |              0.969 |           0.964 |         0.964    |         0.964 |  0.964    | {'criterion': 'entropy', 'max_depth': None, 'min_samples_leaf': 5}     |
| SVM (polynomial kernel)  |              0.916 |           0.916 |         0.933333 |         0.896 |  0.914286 | {'svc__C': 1, 'svc__degree': 3, 'svc__gamma': 1}                       |
| SVM (linear kernel)      |              0.904 |           0.91  |         0.905138 |         0.916 |  0.910537 | {'svc__C': 1}                                                          |
| SVM (sigmoid kernel)     |              0.896 |           0.89  |         0.882353 |         0.9   |  0.891089 | {'svc__C': 0.1, 'svc__gamma': 0.1}                                     |

## Figures
- Training set plot: `train_dataset.png`
- Test set plot: `test_dataset.png`
- Accuracy comparison plot: `model_accuracy.png`

## Decision Tree
- Best cross-validation accuracy: 0.9690
- Test accuracy: 0.9640
- Test precision: 0.9640
- Test recall: 0.9640
- Test F1-score: 0.9640
- Best parameters: `{'criterion': 'entropy', 'max_depth': None, 'min_samples_leaf': 5}`

### Classification Report
```text
              precision    recall  f1-score   support

           0     0.9640    0.9640    0.9640       250
           1     0.9640    0.9640    0.9640       250

    accuracy                         0.9640       500
   macro avg     0.9640    0.9640    0.9640       500
weighted avg     0.9640    0.9640    0.9640       500
```
### Confusion Matrix
```text
[[241   9]
 [  9 241]]
```

## AdaBoost + Decision Tree
- Best cross-validation accuracy: 0.9750
- Test accuracy: 0.9720
- Test precision: 0.9797
- Test recall: 0.9640
- Test F1-score: 0.9718
- Best parameters: `{'estimator__max_depth': 2, 'learning_rate': 1.0, 'n_estimators': 200}`

### Classification Report
```text
              precision    recall  f1-score   support

           0     0.9646    0.9800    0.9722       250
           1     0.9797    0.9640    0.9718       250

    accuracy                         0.9720       500
   macro avg     0.9721    0.9720    0.9720       500
weighted avg     0.9721    0.9720    0.9720       500
```
### Confusion Matrix
```text
[[245   5]
 [  9 241]]
```

## SVM (linear kernel)
- Best cross-validation accuracy: 0.9040
- Test accuracy: 0.9100
- Test precision: 0.9051
- Test recall: 0.9160
- Test F1-score: 0.9105
- Best parameters: `{'svc__C': 1}`

### Classification Report
```text
              precision    recall  f1-score   support

           0     0.9150    0.9040    0.9095       250
           1     0.9051    0.9160    0.9105       250

    accuracy                         0.9100       500
   macro avg     0.9101    0.9100    0.9100       500
weighted avg     0.9101    0.9100    0.9100       500
```
### Confusion Matrix
```text
[[226  24]
 [ 21 229]]
```

## SVM (RBF kernel)
- Best cross-validation accuracy: 0.9790
- Test accuracy: 0.9720
- Test precision: 0.9720
- Test recall: 0.9720
- Test F1-score: 0.9720
- Best parameters: `{'svc__C': 10, 'svc__gamma': 'scale'}`

### Classification Report
```text
              precision    recall  f1-score   support

           0     0.9720    0.9720    0.9720       250
           1     0.9720    0.9720    0.9720       250

    accuracy                         0.9720       500
   macro avg     0.9720    0.9720    0.9720       500
weighted avg     0.9720    0.9720    0.9720       500
```
### Confusion Matrix
```text
[[243   7]
 [  7 243]]
```

## SVM (polynomial kernel)
- Best cross-validation accuracy: 0.9160
- Test accuracy: 0.9160
- Test precision: 0.9333
- Test recall: 0.8960
- Test F1-score: 0.9143
- Best parameters: `{'svc__C': 1, 'svc__degree': 3, 'svc__gamma': 1}`

### Classification Report
```text
              precision    recall  f1-score   support

           0     0.9000    0.9360    0.9176       250
           1     0.9333    0.8960    0.9143       250

    accuracy                         0.9160       500
   macro avg     0.9167    0.9160    0.9160       500
weighted avg     0.9167    0.9160    0.9160       500
```
### Confusion Matrix
```text
[[234  16]
 [ 26 224]]
```

## SVM (sigmoid kernel)
- Best cross-validation accuracy: 0.8960
- Test accuracy: 0.8900
- Test precision: 0.8824
- Test recall: 0.9000
- Test F1-score: 0.8911
- Best parameters: `{'svc__C': 0.1, 'svc__gamma': 0.1}`

### Classification Report
```text
              precision    recall  f1-score   support

           0     0.8980    0.8800    0.8889       250
           1     0.8824    0.9000    0.8911       250

    accuracy                         0.8900       500
   macro avg     0.8902    0.8900    0.8900       500
weighted avg     0.8902    0.8900    0.8900       500
```
### Confusion Matrix
```text
[[220  30]
 [ 25 225]]
```

## Result Discussion
- The best test performer is **AdaBoost + Decision Tree** with accuracy 0.9720.
- The weakest test performer is **SVM (sigmoid kernel)** with accuracy 0.8900.
- This dataset is a noisy, curved and non-linearly separable 3D moon structure, so models that can capture smooth non-linear boundaries usually have an advantage.
- A single decision tree can carve highly irregular boundaries, but it is sensitive to local noise and axis-aligned splits are not a natural fit for crescent-shaped manifolds.
- AdaBoost improves over one tree because multiple weak trees can focus on difficult regions, reducing bias while keeping enough flexibility to follow the moon-shaped boundary.
- SVM performance depends strongly on the kernel. Linear SVM is limited to one global hyperplane, so it cannot model the moon geometry well.
- Non-linear SVM kernels such as RBF and polynomial can map the data into richer feature spaces, which better matches the curved class structure in this task.
- The sigmoid kernel often behaves less stably on this kind of problem, so it may underperform compared with RBF or polynomial kernels.