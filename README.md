## National Institute of Technology Karnataka - Atul Pandey
This is a solo team
Name: Atul Pandey
Degree:  M.Tech(Research)
Department: Department of Information Technology

## Problem Statement: Quantum Detective: Cracking Financial Anomalies 

**Abstract:**

Financial transactions generate massive and complex datasets where unusual patterns or anomalies, such as fraudulent activities or errors, can easily remain hidden. Traditional machine learning models often face challenges with these datasets due to their size and complexity, resulting in slower processing times and less accurate results. Taking advantage of quantum computing, which uses the principle of quantum mechanics. In this hackathon, I proposed a method to detect fraud or not fraud using the quantum neural network (QNN) and Quantum Support Vector Machine (QSVM). However, the performance of these models is not up to the mark; therefore, to improve the performance, I have proposed a hybrid classical-quantum model. The performance of the hybrid model is near that of the classical Artificial Neural Network (ANN) and Support Vector Machine (SVM). I used pennylane to implement the quantum models.

**Dataset**: The dataset used in the hackathon is **Credit Card Transaction Anomalies**. This is available in the kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud. It contains two classes, 0 and 1. 0 represents Fraud, and 1 represents not a fraud. 

No Frauds: 284315

Frauds: 492

The above data shows that the dataset is highly imbalanced dataset. Imbalanced handling techniques and model training are discussed in the methodology. 

**Methodology** 

1. Perform the Exploratory data analysis (EDA) as well as data preprocessing using the reference [1].
   
          EDA includes: Check for Null values, mean, standard deviation, class distributions, Feature distribution (Transaction Amount, Transaction Time)
          Data Preprocessing: 1. Scaling Column Amount and Time
                                   Financial transactions (amounts) often have extreme outliers (e.g., very large purchases), so RobustScaler helps minimize the effect of these outliers while still scaling the 
                                   data appropriately.
                                   Time with StandardScaler: Time differences between transactions may not have extreme outliers and are likely to follow a relatively normal distribution, so StandardScaler is a 
                                   good choice to standardize these values.
                               2. To handle the imbalance, resampling techniques are used. They are of two type
                                         1. Undersampling - random undersampling. This technique reduces the number of instances in the majority class, which will be helpful in training the quantum model because                                            of the lesser number of data.
   
                                         2. Oversampling - SMOTE. This technique generates synthetic examples for the minority class, which will increase the data size ( number of rows). Due to this training                                                   quantum model takes too much time. Even if I used it, ultimately, I have to reduce it for training.
    
                              3.  Handling outlier
   
                              4. Splitting the dataset into train and test
   
 3. Perform training for the classical model ( SVM, ANN), Quantum Model (QNN, QSVM), and Hybrid Classical-Quantum Model
    
    **Note**: Training of the quantum models is performed using the two different types of embedding techniques, i.e. Angle Embedding, Amplitude Embedding
    
 5. The performance of the models is evaluated on the basis of the test dataset. Accuracy, Precision, Recall, F1-Score, and Roc-curve are used for the performance evaluation.

**Conclusion**

In this hackathon, we can see that the highly imbalanced nature of the data affects the performance of the model, either classical or quantum; the pdf of model performance attached in the repo shows that all classical model (SVM and ANN) shows better results than all quantum models( QNN, QSVM, Hybrid model). However, the QSVm and the Hybrid model (with angle embedding) show near about results to classical. This happened because of the Hilbert space used by the quantum kernel and the feature extracted by the classical model, which is then fed to the quantum.   

 **Note**:   
 1. Some of the code files run on the server, and some on the Google Colab. This has happened because of the over usage of Google Colab GPU

 2. The links in the references helped me write quantum code.

 3. All Neural network models, either classical or quantum, are run on the same parameters: epoch =50, batch_size = 40, learning_rate=0.005 

 **Generative AI Tools**: Chat GPT is used to correct the code error for classical and Quantum.

## Instructions on running your project
1. Install the library mentioned in the requirement.txt
2. Connect your system to GPU - Google Colab can be
3. Training the quantum program takes 30 min - 60 min


## References
 1. https://www.kaggle.com/code/rakeshjampa/unmasking-fraud-on-credit-card-fraud-detection
 2. https://pennylane.ai/qml/demos/tutorial_qnn_module_tf/
 3. https://docs.pennylane.ai/en/stable/introduction/templates.html
 4. https://pennylane.ai/plugins/
 5. https://pennylane.ai/qml/demos/tutorial_kernel_based_training/
