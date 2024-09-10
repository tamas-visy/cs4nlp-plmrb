import os
import pandas as pd
import json


def demographic_parity(df, gender_column='group', prediction_column='pred_label'):
    mean_prediction = df.groupby(gender_column)[prediction_column].mean()
    return mean_prediction


def equalized_odds(df, gender_column='group', true_column='TrueSentiment', prediction_column='pred_label'):
    ez_o = df.groupby(gender_column)[prediction_column].mean()
    return ez_o


def relative_sentiment_change(df, mask_prediction_column='pred_label_with_mask', gender_column='group',
                              true_column='TrueSentiment', prediction_column='pred_label'):
    changes = []
    for _, row in df.iterrows():
        original_sentiment = row[prediction_column]
        mask_sentiment = row[mask_prediction_column]
        change = mask_sentiment - original_sentiment
        changes.append((row[gender_column], change))
    return pd.DataFrame(changes, columns=[gender_column, 'Relative Sentiment Change'])


def evaluate(evalDataPath):
    """Calculates metrics based on the real and predicted sentiments"""

    # FOR LABELS
    evalData = pd.read_csv(evalDataPath)
    evalData['pred_label_neg_prob'] = 1 - evalData['pred_label_pos_prob']
    data_negative = evalData[evalData['label'] == -1]
    data_positive = evalData[evalData['label'] == 1]
    data_binary = evalData[evalData['label'] != 0]
    data_neutral = evalData[evalData['label'] == 0]

    # Accuracy
    correct_predictions = (data_binary['label'] == data_binary['pred_label']).sum()
    acc = correct_predictions / len(data_binary)

    # Demographic Parity
    dp = demographic_parity(data_binary, gender_column='group', prediction_column='pred_label')

    # Equalized Odds
    ez_o_neg = equalized_odds(data_negative, gender_column='group', true_column='label', prediction_column='pred_label')
    ez_o_pos = equalized_odds(data_positive, gender_column='group', true_column='label', prediction_column='pred_label')

    # Relative Sentiment Change for neutral examples
    relative_changes = relative_sentiment_change(data_neutral, mask_prediction_column='pred_label_with_mask',
                                                 gender_column='group',
                                                 true_column='label', prediction_column='pred_label')
    average_changes = relative_changes.groupby('group')['Relative Sentiment Change'].mean()

    # FOR PROBS
    evalData = pd.read_csv(evalDataPath)
    data_negative = evalData[evalData['label'] == -1]
    data_positive = evalData[evalData['label'] == 1]
    data_binary = evalData[evalData['label'] != 0]
    data_neutral = evalData[evalData['label'] == 0]

    # Demographic Parity
    dp_prob = demographic_parity(data_binary, gender_column='group', prediction_column='pred_label_pos_prob')

    # Equalized Odds
    ez_o_neg_prob = equalized_odds(data_negative, gender_column='group', true_column='label',
                                   prediction_column='pred_label_pos_prob')
    ez_o_pos_prob = equalized_odds(data_positive, gender_column='group', true_column='label',
                                   prediction_column='pred_label_pos_prob')

    # Relative Sentiment Change for neutral examples
    relative_changes_prob = relative_sentiment_change(data_neutral,
                                                      mask_prediction_column='pred_label_pos_prob_with_mask',
                                                      gender_column='group',
                                                      true_column='label', prediction_column='pred_label_pos_prob')
    average_changes_prob = relative_changes_prob.groupby('group')['Relative Sentiment Change'].mean()

    return acc, dp, ez_o_neg, ez_o_pos, average_changes, dp_prob, ez_o_neg_prob, ez_o_pos_prob, average_changes_prob


def read_and_format_accuracies(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    accuracies = {}
    for line in lines:
        metric, value = line.strip().split(':')
        accuracies[metric.strip()] = f"{float(value) * 100:.2f}"

    return accuracies


def traverse_and_evaluate(data_dir):
    results = {}
    for lm_folder in os.listdir(data_dir):
        # if lm_folder=='GloveLanguageModel':
        #     continue
        print("Inside", lm_folder)
        lm_path = os.path.join(data_dir, lm_folder)
        for probe_folder in os.listdir(lm_path):
            print("\tInside", probe_folder)
            probe_path = os.path.join(lm_path, probe_folder)
            for layer_folder in os.listdir(probe_path):
                print("\t\tInside", layer_folder)
                layer_path = os.path.join(probe_path, layer_folder)
                for root, dirs, files in os.walk(layer_path):
                    for file in files:
                        if file == 'test_data_evaluated.csv':
                            accuracy_file = os.path.join(root, 'accuracy.txt')
                            accuracies = read_and_format_accuracies(accuracy_file)

                            lm = lm_folder
                            ml_model = probe_folder
                            layer = layer_folder

                            test_preds_probs_path = os.path.join(root, file)
                            acc, dp, ez_o_neg, ez_o_pos, avg_changes, dp_prob, ez_o_neg_prob, ez_o_pos_prob, avg_changes_prob = evaluate(
                                test_preds_probs_path)

                            if lm not in results:
                                results[lm] = {}
                            if ml_model not in results[lm]:
                                results[lm][ml_model] = {}
                            if layer not in results[lm][ml_model]:
                                results[lm][ml_model][layer] = {}

                            results[lm][ml_model][layer] = {
                                'Training Accuracy': accuracies['Training Accuracy'],
                                'Validation Accuracy': accuracies['Validation Accuracy'],
                                'Test Accuracy': acc,
                                'Demographic Parity (Label)': dp.to_dict(),
                                'Equalized Odds Negative (Label)': ez_o_neg.to_dict(),
                                'Equalized Odds Positive (Label)': ez_o_pos.to_dict(),
                                'Average Relative Sentiment Change (Label)': avg_changes.to_dict(),
                                'Demographic Parity (Prob)': dp_prob.to_dict(),
                                'Equalized Odds Negative (Prob)': ez_o_neg_prob.to_dict(),
                                'Equalized Odds Positive (Prob)': ez_o_pos_prob.to_dict(),
                                'Average Relative Sentiment Change (Prob)': avg_changes_prob.to_dict(),
                            }

    return results


def main(data_dir, **kwargs):
    """
    This script is designed to evaluate machine learning models by calculating various metrics from CSV files
    containing prediction results. It begins by importing necessary libraries such as `os`, `pandas`, and `json`. The
    script defines functions for calculating key metrics: `demographic_parity`, `equalized_odds`,
    and `relative_sentiment_change`, which compute the mean predictions, equalized odds, and sentiment changes,
    respectively, grouped by gender.

    The `evaluate` function reads an evaluation data CSV file, processes it to calculate the defined metrics for both
    labels and probabilities, and returns these metrics. It first calculates the negative probabilities and separates the
    data into negative, positive, binary, and neutral subsets. Accuracy is calculated by comparing true and predicted
    labels. The function also calculates demographic parity and equalized odds for negative and positive sentiments,
    and the average relative sentiment change for neutral examples.

    The `read_and_format_accuracies` function reads accuracy values from a text file and formats them as percentages. The
    `traverse_and_evaluate` function navigates through a directory structure to locate `test_data_evaluated.csv` files,
    read the corresponding `accuracy.txt` files, and apply the `evaluate` function to compute the metrics. It organizes
    the results in a nested dictionary structure based on language model, machine learning model, and layer.

    Finally, the script sets the root data directory and invokes the `traverse_and_evaluate` function to process the
    evaluation files, saving the results as a JSON file for further analysis and printing them for verification. This
    comprehensive approach ensures thorough evaluation and systematic organization of results across different models and
    layers.

    # --- #
    kwargs is supposed to catch `constant_csv_path`
    """
    # pred_label_pos_prob, pred_label_pos_prob_with_mask also exist

    # Traverse the directory structure and evaluate each test_preds_probs.csv file
    results = traverse_and_evaluate(data_dir)
    with open("IDidAThing.json", "w") as f:
        json.dump(results, f)
    # Printing the results for verification
    import pprint

    pprint.pprint(results)


if __name__ == '__main__':
    main()
