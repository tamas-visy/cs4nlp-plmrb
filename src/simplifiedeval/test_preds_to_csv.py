import os
import pandas as pd


def process_csv(test_preds_probs_path, output_test_preds_probs_path, constant_csv_path):
    # Read the test_preds_probs.csv file
    test_preds_df = pd.read_csv(test_preds_probs_path)

    # Extract the columns pos_probs1 and pos_probs2
    try:
        pos_probs1 = test_preds_df['pos_prob1']
        pos_probs2 = test_preds_df['pos_prob2']
    except:
        pos_probs1 = test_preds_df['prob1']
        pos_probs2 = test_preds_df['prob2']

    # Read the constant CSV file
    constant_df = pd.read_csv(constant_csv_path)
    constant_df.rename(columns={'input_neutral': 'input_with_mask'}, inplace=True)

    # Add the new columns to the constant DataFrame
    constant_df['pred_label_pos_prob'] = pos_probs2
    constant_df['pred_label_pos_prob_with_mask'] = pos_probs1

    # Compute the pred_label and pred_label_with_mask columns
    constant_df['pred_label'] = constant_df['pred_label_pos_prob'].apply(lambda x: 1 if x >= 0.5 else -1)
    constant_df['pred_label_with_mask'] = constant_df['pred_label_pos_prob_with_mask'].apply(
        lambda x: 1 if x >= 0.5 else -1
    )

    # Save the modified DataFrame back to the original location
    constant_df.to_csv(output_test_preds_probs_path, index=False)


def traverse_and_process(data_dir, constant_csv_path):
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
                        if file == 'test_preds_probs.csv':
                            test_preds_probs_path = os.path.join(root, file)
                            output_test_preds_probs_path = os.path.join(layer_path, 'test_data_evaluated.csv')
                            process_csv(test_preds_probs_path, output_test_preds_probs_path, constant_csv_path)


def main():
    """
    The script is designed to process CSV files within a specified directory structure by identifying and manipulating
    `test_preds_probs.csv` files. Initially, it reads these files to extract probability columns `pos_prob1` and
    `pos_prob2`, or alternatively `prob1` and `prob2`, if the former are not found. These probabilities are then
    integrated into a constant CSV file (`generated_eval_dataset_with_mask.csv`) after renaming one of its columns from
    `input_neutral` to `input_with_mask`. The script adds new columns `pred_label_pos_prob` and
    `pred_label_pos_prob_with_mask` to the constant DataFrame, which store the extracted probabilities. Additionally,
    it computes the `pred_label` and `pred_label_with_mask` columns by applying a threshold of 0.5 to the probabilities,
    assigning a value of 1 if the probability is equal to or greater than 0.5, and -1 otherwise. The modified DataFrame,
    which now contains the augmented data, is saved back to the directory as `test_data_evaluated.csv`. The script
    features a `process_csv` function that handles the reading, processing, and writing of CSV files,
    and a `traverse_and_process` function that recursively navigates the directory structure, applying `process_csv` to
    each relevant `test_preds_probs.csv` file it encounters. This approach ensures that all CSV files within the given
    directory are systematically processed and updated according to the defined logic.
    """
    # 1 is NEUTRAL, 2 is regular!

    # Define the constant CSV file path for test data
    constant_csv_path = '/content/drive/MyDrive/cs4nlp-plmrb-main/data/processed/generated_eval_dataset_with_mask.csv'
    # Define the root data directory
    data_dir = '/content/drive/MyDrive/outputs_Hf_shuffle'

    # Traverse the directory structure and process each test_preds_probs.csv file
    traverse_and_process(data_dir, constant_csv_path)


if __name__ == '__main__':
    main()
