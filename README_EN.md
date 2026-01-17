# TensorFlow 2.0 Learning Project

A comprehensive project for learning TensorFlow 2.0, including fundamental syntax explanations, code examples, and practical use cases. This project aims to help developers systematically master TensorFlow 2.0 core concepts and practical applications from basics to advanced topics.

[中文文档](README.md) | English

## 📚 Project Structure

```
TensorFlow2Demo/
├── 01_Basics/                     # Fundamental concepts
│   ├── tensorflow_basics.py        # Tensors, variables, autodiff, etc.
│   └── eager_execution_demo.py    # Eager execution demo
│
├── 02_Neural_Networks/            # Neural network basics
│   ├── neural_networks_basics.py  # Layers, activations, loss functions
│   └── custom_layers_models.py    # Custom layers and models
│
├── 03_Data_Processing/             # Data processing
│   ├── data_processing_basics.py  # Data processing fundamentals
│   └── tfdataset_examples.py      # tf.data API examples
│
├── 04_Model_Training/             # Model training and evaluation
│   └── model_training_basics.py   # Training, evaluation, save/load
│
├── 05_Practical_Cases/            # Practical examples
│   ├── image_classification.py    # Image classification
│   └── nlp_text_classification.py # Text classification
│
├── data/                          # Data storage
├── models/                        # Model storage
├── notebooks/                     # Jupyter notebooks
├── 06_Utils/                      # Utility functions
├── tests/                         # Unit tests
├── requirements.txt               # Project dependencies
├── run_tests.py                   # Test runner script
└── README.md                      # Project documentation
```

## 🚀 Quick Start

### Requirements

- Python 3.7+
- TensorFlow 2.15.0+
- NumPy 1.24.0+
- matplotlib 3.8.0+
- pandas 2.1.0+
- scikit-learn 1.3.0+

### Installation

```bash
pip install -r requirements.txt
```

Or install main dependencies manually:

```bash
pip install tensorflow numpy matplotlib pandas scikit-learn jupyter Pillow
```

### Running Tests

The project includes comprehensive unit tests to ensure code quality:

```bash
# Run all tests
python run_tests.py

# Run specific module tests
python run_tests.py -m 01_basics

# View test coverage
pytest --cov=. --cov-report=html
```

## 📖 Learning Path

### 1. Basics (01_Basics/)

Start with TensorFlow 2.0 fundamental concepts:

- **Tensors**: Learn TensorFlow's basic data structure
- **Eager Execution**: Understand TensorFlow 2.0's execution mode
- **Variables and Autodiff**: Master variable definition and gradient computation
- **Function Decorators**: Learn tf.function usage

Run examples:
```python
python 01_Basics/tensorflow_basics.py
python 01_Basics/eager_execution_demo.py
```

### 2. Neural Networks (02_Neural_Networks/)

Learn basic components for building neural networks:

- **Keras API**: Master Sequential, Functional, and Subclassing APIs
- **Network Layers**: Understand various layer types and usage
- **Activation Functions**: Learn characteristics of different activations
- **Loss Functions and Optimizers**: Choose appropriate loss functions and optimization strategies

Run examples:
```python
python 02_Neural_Networks/neural_networks_basics.py
python 02_Neural_Networks/custom_layers_models.py
```

### 3. Data Processing (03_Data_Processing/)

Master TensorFlow's data processing capabilities:

- **tf.data API**: Efficient data pipelines
- **Data Preprocessing**: Standardization, normalization, encoding
- **Data Augmentation**: Improve model generalization
- **Performance Optimization**: Parallel processing, caching, prefetching

Run examples:
```python
python 03_Data_Processing/data_processing_basics.py
python 03_Data_Processing/tfdataset_examples.py
```

### 4. Model Training (04_Model_Training/)

Learn complete model training workflow:

- **Model Compilation**: Optimizer, loss function, and metrics setup
- **Training Techniques**: Callbacks, regularization, learning rate scheduling
- **Model Evaluation**: Accuracy, precision, recall metrics
- **Model Persistence**: Save and load models

Run examples:
```python
python 04_Model_Training/model_training_basics.py
```

### 5. Practical Cases (05_Practical_Cases/)

Consolidate knowledge through practical examples:

- **Image Classification**: CNN, transfer learning, ResNet
- **Text Classification**: Word embeddings, RNN, LSTM, CNN

Run examples:
```python
python 05_Practical_Cases/image_classification.py
python 05_Practical_Cases/nlp_text_classification.py
```

## 💡 Key Features

### 1. Practical

Each module contains theoretical explanations and practical code examples that can be run directly.

### 2. Progressive Learning

Learn TensorFlow 2.0 progressively from basic concepts to advanced applications.

### 3. Best Practices

Includes code optimization, performance tuning, and best practices for real-world development.

### 4. Rich Examples

Covers various application scenarios including image processing, text processing, and time series.

## 🛠️ Tools and Techniques

### 1. Performance Optimization

- Build efficient data pipelines using tf.data API
- Accelerate training with parallel processing
- Proper use of caching and prefetching

### 2. Debugging Tips

- Eager execution for easy debugging
- TensorBoard visualization
- Custom callbacks for training monitoring

### 3. Model Deployment

- Model saving and loading
- SavedModel format
- Convert to TFLite (mobile)

## 🎯 Learning Objectives

Through this project, you will be able to:

1. **Master TensorFlow 2.0 core concepts**
2. **Proficiently use Keras API to build models**
3. **Efficiently process various types of data**
4. **Implement common deep learning tasks**
5. **Optimize model performance and training process**

## 📊 Example Code

Here's a simple TensorFlow 2.0 example:

```python
import tensorflow as tf

# Create model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Prepare data
import numpy as np
X = np.random.random((1000, 10))
y = np.random.randint(0, 2, (1000, 1))

# Train model
history = model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)
```

## 🧪 Unit Tests

This project includes comprehensive unit tests to ensure code quality and functionality.

### Test Coverage

- ✅ **Basics Tests**: Tensor operations, variables, autodiff
- ✅ **Neural Networks Tests**: Keras API, layers, activations
- ✅ **Data Processing Tests**: tf.data API, preprocessing, augmentation
- ✅ **Model Training Tests**: Compilation, training, evaluation, save/load
- ✅ **Utility Tests**: Visualization, parameter statistics
- ✅ **Integration Tests**: End-to-end workflows

### Quick Test Run

```bash
# Run all tests
python run_tests.py

# Run specific module tests
python run_tests.py -m 01_basics
python run_tests.py -m 02_neural_networks

# Use pytest
pytest

# Generate coverage report
pytest --cov=. --cov-report=html
```

## 🔗 Related Resources

- [TensorFlow Official Documentation](https://www.tensorflow.org/)
- [Keras Official Documentation](https://keras.io/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [TensorFlow Hub](https://tfhub.dev/)

## 📝 Learning Recommendations

1. **Progressive Learning**: Follow module order to build a solid foundation
2. **Hands-on Practice**: Run each example, modify parameters to observe results
3. **Deep Understanding**: Don't just use APIs, understand the underlying principles
4. **Project Practice**: Apply learned knowledge to real projects

## 🤝 Contributing

Welcome to submit issue reports, improvement suggestions, or contribute code directly!

## 📄 License

This project is licensed under the MIT License.

---

**Happy Learning!**

Start your TensorFlow 2.0 learning journey!
