# Project Overview
This project provides a structured approach to understand and predict user scores based purely on their interaction patterns, employing advanced machine learning techniques to achieve meaningful insights.

## Data Analysis
The data analysis is conducted on the log file. The log file contains valuable information about the user's actions and interactions, which are used to predict their scores.



## Feature Engineering
Feature engineering is performed to transform raw data into features that better represent the underlying problem to the predictive models, resulting in improved model accuracy.

## Sequence Modeling
Sequence modeling involves using the BERT model to predict the sequence of actions. The BERT model is a transformer-based machine learning technique for natural language processing pre-training. 

In this project, we use BERT to generate special tokens that represent the sequence of actions. These tokens are then used to predict the user's score.

### Data
The data used in this project is stored in the [data](data/) directory. It includes various files such as data.xlsx, group_a_raw.csv, Groupa_scores.xlsx, Groupb_scores.xlsx, integration_log_group_a.xlsx, and integration_log_group_b.xlsx.
### Model Training and Evaluation
The model is trained on the sequences and scores data, and its performance is evaluated on the test set.

### Weights and Biases Integration
We use Weights and Biases for experiment tracking, model optimization, and dataset versioning. It helps us to keep track of our experiments, visualize our results, and share our findings with others.

# Part 1: Analysis of User Score Prediction

This project leverages actual user interaction logs within an educational environment to predict scores. By capturing the sequence and nature of user actions without direct knowledge of the answers chosen or their correctness, the model aims to infer user scores based on behavioral patterns.

Refer to [Part1](/DataAnalysis) for more information.


# Part 2: Sequential Modeling with BERT

This project aims to predict user scores based on sequences of their interaction logs with a quiz system, without knowledge of the correct answers or the choices made by the user. The primary challenge is to infer the score from patterns in user behavior during the quiz.

Refer to [Part2](/SeqModel) for more information.




