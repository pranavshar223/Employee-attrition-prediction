import pandas as pd

def preprocess_input(input_df, feature_names):
    """
    Preprocess new input data to match training format.
    """

    # One-hot encoding
    input_df = pd.get_dummies(input_df)

    # Align columns with training features
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    return input_df