# Part 2: User Interaction Log Score Prediction with BERT

## Overview
This project aims to predict user scores based on sequences of their interaction logs with a quiz system, without knowledge of the correct answers or the choices made by the user. The primary challenge is to infer the score from patterns in user behavior during the quiz.

## Setup
To run this project, ensure you have Python installed and the required packages, which can be installed via pip:

```bash
cd AITutor_SeqModeling
pip install virtualenv
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

If you have GPU available:

```bash
conda uninstall pytorch
pip uninstall torch
pip uninstall torch # run this command twice
```

Then look for pytorch [local installation guide](https://pytorch.org/get-started/locally/) based on your OS and Cuda version. Here is an example for pip in Linux Cuda 11.8

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir
```



To run the main script:

```bash
python main.py
```

You can also use the Jupyter notebook file for quick testing. 

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
