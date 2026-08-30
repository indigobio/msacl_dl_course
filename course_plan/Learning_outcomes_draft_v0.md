Course Schedule

Segment 1 : Sunday 09:00 - 12:20 (3.00 contact hrs)
Segment 2 : Sunday 14:40 - 18:00 (3.00 contact hrs)
Segment 3 : Monday 09:00 - 12:20 (3.00 contact hrs)
Segment 4 : Monday 14:40 - 18:00 (3.00 contact hrs)
Segment 5 : Tuesday 09:00 - 12:20 (3.00 contact hrs)

Total Contact Hours: 15.00
(Ten-minute breaks occur after each full instructional hour when another hour follows. Breaks are excluded from contact hour calculations.)

---------------

Pre-requisites

Data Science 203 (or equivalent experience)

Overview

Deep learning has transformed scientific research and industry applications, powering breakthroughs from protein structure prediction (AlphaFold) to medical image analysis and intelligent systems. In fact, the AI revolution we're experiencing today, including large language models like ChatGPT, is built entirely on deep learning.

But ... what exactly is deep learning? How is it different from traditional machine learning? How can you apply it to your data without a PhD in mathematics/computer science?

In this course, we will demystify deep learning and provide a practical introduction focused on intuition over mathematics. We'll cover fundamental of neural networks, popular architectures, and practical applications, especially in mass spectrometry.

We will include hands-on instruction on building and training neural networks using Python and modern deep learning frameworks (PyTorch), so familiarity with basic Python programming concepts is desirable but advanced math knowledge is NOT required.

Topics Covered


Fundamentals of neural networks
Comparison of deep learning and traditional machine learning
How neural networks learn
Common practices in network training
Popular neural network architectures
Transformers and attention mechanisms
Different learning paradigms (unsupervised, self-supervised)
Practical applications in mass spectrometry
Objectives

At the conclusion of this short course, the participant will be able to:

Explain principles of deep learning in accessible terms
Build and train basic neural networks in PyTorch
Understand modern deep learning architectures and their use cases
Apply appropriate network architectures to different types of data
Make informed decisions about when deep learning is appropriate for their research



Lecture by Lecture Learning Outcomes (Each lecture is a one-hour sessions)
---

1. Mathematics Fundamentals
   - Basic Linear Algebra and Calculus Refresher
    - Matrix Multiplication
    - Activation Functions
    - Derivatives and Gradients
	- (Assessment: Short quiz with small math problems via handouts or a online platform)
   - MLP and PyTorch Basics
	- Structure of a simple neural network
	- Forward propagation
	- A simple example of a neural network for classification on PyTorch. Demo with walk through of the code.
		- Code exercise for basic Python and PyTorch examples
		- A quick assessment of reading Python and PyTorch code (without backpropagation yet)

2. Backpropagation and Training
   - Backpropagation and Gradient Descent
	- How neural networks learn
	- Loss functions and optimization
	- Learning Rate
	- (Assessment: Short quiz on backpropagation math with small calculation problems)
   - Training Neural Networks in PyTorch, continued from the previous example code
	- Implementing backpropagation in PyTorch
	- Training a simple neural network on a dataset
	- Code exercise for training a neural network with PyTorch

3. (Lab) Train A MLP on a Real Dataset
   - Colab Setup
   - Training a Multi-Layer Perceptron (MLP) on a real dataset
	- What is Google Colab and how to use it for deep learning
    - (Use a online platform like Google Colab for code completion, for each to be completed part, provide options on a paper sheet?)
	- Data preprocessing and normalization
	- Splitting data into training and test sets
	- Training the MLP and evaluating its performance

4. General Problems
   - Better Loss Functions
	- Cross-entropy loss for classification tasks
   - From Gradient Descent to Stochastic Gradient Descent
     - Mini-batch training
	 - Touch on Adam optimizer and other optimizers
   - Deep is Better but Gradients are Harder
	 - Vanishing and exploding gradients
   - Overfitting and Underfitting
	- Regularization techniques (Dropout, L2 regularization)
	- Early stopping
	- (Assessment: Short quiz on overfitting and underfitting concepts)
   - Hyperparameter Tuning
	- Learning rate, batch size, number of epochs
	- Grid search and random search
	- Code exercise for hyperparameter tuning in PyTorch

5. Convolutional Neural Networks (CNNs)
   - Introduction to CNNs
	- Convolutional layers, pooling layers, and fully connected layers
	- Common architectures (LeNet, AlexNet, VGG, ResNet)
	- Resnet and skip connections
	- (Assessment: Short quiz on CNN concepts and architectures, math to compute filter size)

6. (Lab) Implementing a CNN in PyTorch
	- Building a simple CNN for image classification, from a clinical image dataset
	- Data augmentation techniques
	- Hands-on learning on hyperparameter tuning for CNNs
  	  - Batch size
	  - CNN architecture, more CNN layers

7. Sequence Models and Transformers
   - Introduction to sequence models (RNNs, LSTMs)
	- An example in time series
   - Transformers and Attention Mechanisms
    - NLP Basics:
      - Tokenization and embeddings
	- Self-attention and multi-head attention (Heavy visualization and examples)
	- (Assessment: Simple math problem on attention mechanism)

8. Transformers Continued
   - Positional encoding
   - Normalization:
     - Batch normalization
     - Layer normalization
   - Transformer architecture Encoder and Decoder
   - Transfer learning 
	- Retrain prediction heads on pre-trained models

9. (Lab) Implementing a Transformer for NLP Tasks
   - Building a transformer model for a text classification task
   - Fine-tuning pre-trained transformer models (e.g., BERT, GPT)
   - Hands-on learning on hyperparameter tuning for transformers

10. Object Detection and Segmentation
   - Introduction to object detection and segmentation
   - Loss functions for object detection and segmentation
   - Popular architectures (YOLO, Faster R-CNN, Mask R-CNN)
   - Applications in medical imaging and mass spectrometry
   - (Assessment: Short quiz on object detection and segmentation concepts)

11. Different Learning Paradigms
   - Unsupervised Learning
	 - Autoencoders
	 - Variational Autoencoders (VAEs)
   - Self-Supervised Learning
	 - Contrastive learning
	 - Pretext tasks
   - Semi-Supervised Learning
	 - Pseudo-labeling
	 - Consistency regularization
   - (Assessment: Short quiz on different learning paradigms)

12. (Lab) Implementing Supervised object detection (Colab)
   - Building an unsupervised object detection model on a medical imaging dataset
   - Hands-on learning on hyperparameter tuning for object detection

13. Large Language Models (LLMs) and Applications
   - Introduction to LLMs
   - Pre-training and SFT
   - Post-training and fine-tuning
   - (Assessment: Short quiz on LLM concepts and applications)

14. Agentic AI
   - Reinforcement learning approach
     - Introduction to reinforcement learning
	 - Markov Decision Processes (MDPs)
	 - Policy gradients and value-based methods
   - ReAct 
	 - Combining reasoning and acting in LLMs
	 - Applications of ReAct in real-world scenarios
   - (Assessment: Short quiz on agentic AI concepts and applications)

15. (Lab) TBD something with Mass Spectrometry
   - TBD: Implementing a deep learning model for mass spectrometry data analysis
   - Hands-on learning on hyperparameter tuning for mass spectrometry applications
  

