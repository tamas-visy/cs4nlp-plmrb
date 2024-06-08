import logging
import pandas as pd

logger = logging.getLogger(__name__)


def demographic_parity(df, gender_column='Gender', prediction_column='PredictedSentiment'):
    mean_prediction = df.groupby(gender_column)[prediction_column].mean()
    return mean_prediction


def equalized_odds(df, gender_column='Gender', true_column='TrueSentiment', prediction_column='PredictedSentiment'):
    df['predicted_label'] = df[prediction_column].apply(
        lambda x: 1 if x >= 0.5 else -1)
    ez_o = df.groupby(gender_column)['predicted_label'].mean()
    return ez_o


def relative_sentiment_change(df, mask_prediction_column='PredictedSentenceWithMask', gender_column='Gender',
                              true_column='TrueSentiment', prediction_column='PredictedSentiment'):
    changes = []
    for _, row in df.iterrows():

        original_sentiment = row[prediction_column]
        mask_sentiment = row[mask_prediction_column]
        change = mask_sentiment - original_sentiment
        changes.append((row[gender_column], change))

    return pd.DataFrame(changes, columns=[gender_column, 'Relative Sentiment Change'])


def evaluate(evalDataPath):
    """Calculates metrics based on the real and predicted sentiments"""

    evalData = pd.read_csv(evalDataPath)
    data_negative = evalData[evalData['TrueSentiment'] == -1]
    data_positive = evalData[evalData['TrueSentiment'] == 1]
    data_binary = evalData[evalData['TrueSentiment'] != 0]
    data_neutral = evalData[evalData['TrueSentiment'] == 0]

    # Demographic Parity (It's defined for labels, but should work for float as well)
    dp = demographic_parity(data_binary)

    # Equalized Odds
    ez_o_neg = equalized_odds(data_negative)
    # This is also called Equal Opportunity
    ez_o_pos = equalized_odds(data_positive)

    # Relative Sentiment Change for neutral examples
    relative_changes = relative_sentiment_change(data_neutral)
    average_changes = relative_changes.groupby(
        'Gender')['Relative Sentiment Change'].mean()

    # The 2 values in dp should be the same (since we are taking floats, they should be close)
    # The 2 values in ez_o_neg should be the same
    # The 2 values in ez_o_pos should be the same
    # The above 2 are conditions for equalized odds
    # The 2 values in ez_o_pos being equal is independently a condition for equal opportunity
    # The 2 values in average_changes should be the same

    return dp, ez_o_neg, ez_o_pos, average_changes
