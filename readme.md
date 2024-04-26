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

## Project Overview
This project leverages actual user interaction logs within an educational environment to predict scores. By capturing the sequence and nature of user actions without direct knowledge of the answers chosen or their correctness, the model aims to infer user scores based on behavioral patterns.

## Key Points

### Data Description
- **Actions**: A variety of user interactions with the system, such as changing answers, requesting hints, and engaging with educational tools like Streamlit.
- **Features**: Each action is treated both as a standalone feature and as part of a cumulative behavioral profile, including frequency and context of interactions.
- **Score Prediction**: The outcome is the predicted score based on the normalized frequency and sequence of actions, refined through feature engineering to enhance predictive accuracy.

### Approach

#### Feature Engineering
- **Action Count Normalization**: Actions are normalized by the total number of interactions per user to account for varying levels of engagement.
- **One-Hot Encoding**: Problems are encoded to distinguish their impact on user scores.
- **Removal of System Actions**: Actions that do not represent user decisions (e.g., auto-saves, system errors) are excluded from the feature set.

#### Model Development
- **Initial Model**: Starts with basic regression to establish a baseline for prediction accuracy.
- **Feature Optimization**: Incorporates one-hot encoding for problems and selective feature inclusion to improve model performance.
- **Outlier Removal**: Applies the 1.5 IQR method to exclude outliers and improve dataset quality.
- **Correlation Analysis**: Examines relationships between features and scores to refine feature selection.

### Implementation Steps
1. **Data Preprocessing**: Clean and prepare interaction logs for analysis.
2. **Feature Selection and Engineering**: Develop and refine features that accurately represent user interactions.
3. **Model Training and Evaluation**: Train models using engineered features to predict scores; evaluate using RMSE and R-squared metrics.
4. **Iterative Refinement**: Continuously refine the model by incorporating new features and adjusting model parameters.

### Results
- Improved prediction accuracy through advanced feature engineering and model refinement.
- Identification of key features such as problem complexity and interaction types that significantly influence user scores.

### Future Work
- **Feature Expansion**: Explore additional user interactions and temporal patterns for potential inclusion as features.
- **Model Complexity**: Investigate more complex models or ensemble methods to further enhance prediction accuracy.
- **User Group Analysis**: Extend the model to differentiate between user groups (e.g., those with access to additional tools like GPT) and tailor predictions accordingly.

## Conclusion
This project demonstrates the feasibility of using machine learning to predict educational outcomes based solely on user interaction data. Through meticulous feature engineering and model development, we can gain valuable insights into how user behavior correlates with learning success.


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


