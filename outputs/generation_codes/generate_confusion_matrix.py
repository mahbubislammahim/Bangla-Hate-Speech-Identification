import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os

# Define paths (adjust if your structure is different)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
OUTPUT_DIR = os.path.join(BASE_DIR, 'confusion_matrices_plots') # Directory to save plots

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

SUBTASKS = {
    '1A': {
        'test_file': os.path.join(DATA_DIR, 'subtask_1A', 'blp25_hatespeech_subtask_1A_test_with_labels.tsv'),
        'result_file': os.path.join(RESULTS_DIR, 'subtask_1A.tsv'),
        'label_col': 'label',
        'id_col': 'id'
    },
    '1B': {
        'test_file': os.path.join(DATA_DIR, 'subtask_1B', 'blp25_hatespeech_subtask_1B_test_with_labels.tsv'),
        'result_file': os.path.join(RESULTS_DIR, 'subtask_1B.tsv'),
        'label_col': 'label',
        'id_col': 'id'
    },
    '1C': {
        'test_file': os.path.join(DATA_DIR, 'subtask_1C', 'blp25_hatespeech_subtask_1C_test_with_labels.tsv'),
        'result_file': os.path.join(RESULTS_DIR, 'subtask_1C.tsv'),
        'label_col': 'hate_severity',
        'id_col': 'id'
    }
}

def load_labels_and_predictions(subtask_config):
    """
    Loads true labels from the test set and predictions from the results file.
    Handles NaN IDs and labels by converting them to placeholder and 'None' string respectively.
    """
    try:
        # Load true labels
        true_labels_df = pd.read_csv(subtask_config['test_file'], sep='\t')
        # Load predictions
        predictions_df = pd.read_csv(subtask_config['result_file'], sep='\t')

        # Ensure dataframes are not empty
        if true_labels_df.empty or predictions_df.empty:
            print(f"Error: Empty dataframe encountered for {subtask_config['test_file']} or {subtask_config['result_file']}")
            return None, None

        id_col_name = subtask_config['id_col']
        label_col_name = subtask_config['label_col']
        
        # Handle NaN IDs by filling with a placeholder string
        true_labels_df[id_col_name] = true_labels_df[id_col_name].fillna('NaN_id_placeholder_true')
        predictions_df[id_col_name] = predictions_df[id_col_name].fillna('NaN_id_placeholder_pred')

        # Select only the ID and label columns to avoid issues with extra columns
        cols_to_select = [id_col_name, label_col_name]
        
        # Merge the two dataframes on the ID column
        merged_df = pd.merge(true_labels_df[cols_to_select],
                             predictions_df[cols_to_select],
                             on=id_col_name,
                             suffixes=('_true', '_pred'))

        if merged_df.empty:
            print(f"Error: Merge resulted in an empty dataframe. Check ID columns and data for {subtask_config['test_file']} and {subtask_config['result_file']}")
            return None, None
            
        # Extract true and predicted labels, handling NaNs by converting to 'None' string
        true_labels = merged_df[f'{label_col_name}_true'].fillna('None').astype(str)
        predictions = merged_df[f'{label_col_name}_pred'].fillna('None').astype(str)
        
        return true_labels, predictions
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}. Please check paths in SUBTASKS configuration.")
        return None, None
    except KeyError as e:
        print(f"Error: Missing column {e} in one of the files for subtask. Check 'label_col' and 'id_col' in SUBTASKS configuration.")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred while loading data: {e}")
        return None, None

def plot_confusion_matrix(y_true, y_pred, subtask_name, labels=None):
    """
    Computes and plots a colorful confusion matrix.
    Saves the plot to a file.
    """
    if y_true is None or y_pred is None:
        print(f"Cannot plot confusion matrix for {subtask_name} due to data loading issues.")
        return

    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(max(10, len(labels) * 0.5), max(8, len(labels) * 0.5))) # Adjust figure size dynamically
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                annot_kws={"size": 14, "weight": "bold"}) # Adjust font size for potentially many labels
    # plt.title(f'Confusion Matrix for Subtask {subtask_name}', fontsize=16, weight='bold') # Title removed as per user request
    plt.ylabel('True Label', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout() # Adjust layout to make room for labels
    
    output_path = os.path.join(OUTPUT_DIR, f'confusion_matrix_subtask_{subtask_name}.png')
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix for Subtask {subtask_name} saved to {output_path}")
    except Exception as e:
        print(f"Error saving plot for {subtask_name}: {e}")
    plt.close()

def main():
    """
    Main function to generate confusion matrices for all configured subtasks.
    """
    for subtask_name, config in SUBTASKS.items():
        print(f"\nProcessing Subtask {subtask_name}...")
        true_labels, predictions = load_labels_and_predictions(config)
        
        if true_labels is not None and predictions is not None:
            # Get unique sorted labels for the matrix axes from the actual data being compared
            unique_labels = sorted(list(set(true_labels) | set(predictions)))

            if not unique_labels:
                print(f"No labels found to plot for {subtask_name}. Skipping.")
                continue

            print(f"True labels found: {len(true_labels)}")
            print(f"Predictions found: {len(predictions)}")
            print(f"Unique labels for matrix: {unique_labels}")
            
            plot_confusion_matrix(true_labels, predictions, subtask_name, labels=unique_labels)
        else:
            print(f"Skipping confusion matrix generation for Subtask {subtask_name} due to data loading errors.")

if __name__ == '__main__':
    main()
    print("\nConfusion matrix generation process complete.")
    print(f"Plots are saved in the '{OUTPUT_DIR}' directory.")
