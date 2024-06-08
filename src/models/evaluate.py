import logging
import pandas as pd

logger = logging.getLogger(__name__)


def demographic_parity(df, gender_column='Gender', prediction_column='PredictedSentiment'):
    mean_prediction = df.groupby(gender_column)[prediction_column].mean()
    return mean_prediction

def equalized_odds(df, gender_column='Gender', true_column='TrueSentiment', prediction_column='PredictedSentiment'):
    df['predicted_label'] = df[prediction_column].apply(lambda x: 1 if x >= 0.5 else -1)
    tpr = df[df[true_column] == 1].groupby(gender_column)['predicted_label'].mean()
    fpr = df[df[true_column] == -1].groupby(gender_column)['predicted_label'].mean()
    return tpr, fpr

def equal_opportunity(df, gender_column='Gender', true_column='TrueSentiment', prediction_column='PredictedSentiment'):
    df['predicted_label'] = df[prediction_column].apply(lambda x: 1 if x >= 0.5 else -1)
    tpr = df[df[true_column] == 1].groupby(gender_column)['predicted_label'].mean()
    return tpr


def relative_sentiment_change(df, mask_prediction_column = 'PredictedSentenceWithMask', 
                              true_column='TrueSentiment', prediction_column='PredictedSentiment'):
    changes = []
    for _, row in df.iterrows():

        original_sentiment = row[prediction_column]
        mask_sentiment = row[mask_prediction_column]
        change = mask_sentiment - original_sentiment 
        changes.append((row['Gender'], change))
    
    return pd.DataFrame(changes, columns=['Gender', 'Relative Sentiment Change'])


def evaluate(evalDataPath):
    """Calculates metrics based on the real and predicted sentiments"""
    
    evalData = pd.read_csv(evalDataPath)
    data_negative = evalData[evalData['TrueSentiment'] == -1]
    data_positive = evalData[evalData['TrueSentiment'] == 1]
    data_neutral = evalData[evalData['TrueSentiment'] == 0]

    # Demographic Parity (It's only defined for positive, but we can calculate it for negative as well)
    # dp_negative = demographic_parity(data_negative)
    dp_positive = demographic_parity(data_positive)

    # Equalized Odds
    tpr_negative, fpr_negative = equalized_odds(data_negative)
    tpr_positive, fpr_positive = equalized_odds(data_positive)

    # Equal Opportunity
    eo_positive = equal_opportunity(data_positive)

    # Relative Sentiment Change for neutral examples
    relative_changes = relative_sentiment_change(data_neutral)
    average_changes = relative_changes.groupby('Gender')['Relative Sentiment Change'].mean()

    return dp_positive, tpr_positive, fpr_positive, eo_positive, average_changes
