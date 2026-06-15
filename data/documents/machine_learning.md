# Machine Learning Fundamentals

## What is Machine Learning?

Machine learning (ML) is a subset of artificial intelligence (AI) that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.

The process begins with observations or data, such as examples, direct experience, or instruction, to look for patterns in data and make better decisions in the future. The primary aim is to allow computers to learn automatically without human intervention or assistance and adjust actions accordingly.

## Types of Machine Learning

### Supervised Learning

Supervised learning is a type of machine learning where the model is trained on labeled data. The algorithm learns from input-output pairs, where both the input data and the desired output are provided. Common applications include:

- **Classification**: Categorizing emails as spam or not spam, image recognition, medical diagnosis
- **Regression**: Predicting house prices, stock market forecasting, weather prediction

Popular algorithms include Linear Regression, Decision Trees, Random Forests, Support Vector Machines (SVM), and Neural Networks.

### Unsupervised Learning

Unsupervised learning works with unlabeled data, attempting to find hidden patterns or intrinsic structures. The system tries to learn without a teacher. Key techniques include:

- **Clustering**: K-Means, DBSCAN, Hierarchical Clustering
- **Dimensionality Reduction**: PCA (Principal Component Analysis), t-SNE, UMAP
- **Association Rules**: Market basket analysis, recommendation systems

### Reinforcement Learning

Reinforcement learning involves an agent that learns to make decisions by taking actions in an environment to maximize cumulative reward. Key concepts include:

- **Agent**: The learner or decision-maker
- **Environment**: What the agent interacts with
- **Reward Signal**: Feedback from the environment
- **Policy**: Strategy the agent follows

Applications include game playing (AlphaGo, Atari), robotics, autonomous vehicles, and resource management.

## Deep Learning

Deep learning is a subset of machine learning based on artificial neural networks with multiple layers (deep neural networks). These networks can learn complex patterns in large amounts of data.

### Neural Network Architecture

A typical neural network consists of:

1. **Input Layer**: Receives the raw data
2. **Hidden Layers**: Process information through weighted connections
3. **Output Layer**: Produces the final prediction

### Common Architectures

- **Convolutional Neural Networks (CNNs)**: Specialized for image processing and computer vision tasks
- **Recurrent Neural Networks (RNNs)**: Designed for sequential data like text and time series
- **Transformers**: State-of-the-art architecture for natural language processing, powering models like GPT, BERT, and LLaMA
- **Generative Adversarial Networks (GANs)**: Two networks competing to generate realistic synthetic data

## Model Evaluation

Evaluating machine learning models is crucial for understanding their performance:

- **Accuracy**: Percentage of correct predictions
- **Precision**: Ratio of true positives to all predicted positives
- **Recall**: Ratio of true positives to all actual positives
- **F1 Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the Receiver Operating Characteristic curve
- **Mean Squared Error (MSE)**: Average squared difference between predicted and actual values

## Overfitting and Underfitting

**Overfitting** occurs when a model learns the training data too well, including noise, and fails to generalize to new data. Solutions include regularization, dropout, cross-validation, and using more training data.

**Underfitting** happens when a model is too simple to capture the underlying patterns. Solutions include using more complex models, adding features, and reducing regularization.
