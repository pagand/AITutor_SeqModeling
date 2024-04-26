# Part 1: User Interaction Log Score Prediction

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
