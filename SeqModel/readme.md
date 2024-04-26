# Part 2: User Interaction Log Score Prediction with BERT

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
