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


# Part 2: Sequential Modeling for User Score Prediction (BERT)

## Overview
This project aims to predict user scores based on sequences of their interaction logs with a quiz system, without knowledge of the correct answers or the choices made by the user. The primary challenge is to infer the score from patterns in user behavior during the quiz.

## Key Points
- **Actions**: Distinct types of interactions like changing an answer, requesting a hint, visiting educational resources (EDA), and interacting with a chatbot.
- **States**: Representations of the user's quiz attempt at each action log, including time spent, number of hints requested, and other engagement metrics.
- **Labels**: Final scores of students, which are the target predictions of the model.

## Approach
- **Data Representation**: Treat user actions as sequential data, maintaining the temporal order.
- **Feature Engineering**: Convert actions and states into numerical features suitable for machine learning models.
- **Model**: Utilize a Transformer architecture, specifically BERT, known for excellent performance on sequential data.
- **Training**: Data is preprocessed and transformed into features, followed by training the Transformer model.
- **Evaluation**: The model is evaluated using metrics like Mean Squared Error to assess prediction accuracy.

## Implementation Steps
1. **Data Collection and Preprocessing**: Gather and preprocess data into a format compatible with the Transformer model.
2. **Feature Engineering**: Develop comprehensive features that encapsulate diverse aspects of user interactions.
3. **Model Training**: Train the model using the prepared dataset, adjusting parameters as necessary.
4. **Model Evaluation**: Validate the model's performance on a separate test set to ensure its effectiveness.

## Future Enhancements
- Explore ensemble techniques to combine multiple model predictions for improved accuracy.
- Incorporate more granular time-based features and patterns of actions to refine the understanding of user behavior.


