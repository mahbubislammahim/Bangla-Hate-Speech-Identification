#!/usr/bin/env python3
"""
Script to generate a combined results table for research paper
containing accuracy, F1 score, precision, and recall for all subtasks,
plus per-label class-wise results.
"""

import sys
import os
sys.path.append('.')

from scorer.task import evaluate, evaluate_1C, _read_tsv_input_file, _read_gold_labels_file, correct_labels
import pandas as pd
import logging
from sklearn.metrics import classification_report
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def calculate_metrics_for_subtask(pred_file_path, gold_file_path, subtask):
    """
    Calculate metrics for a given subtask
    """
    try:
        # Read predictions and gold labels
        pred_labels = _read_tsv_input_file(pred_file_path)
        gold_labels = _read_gold_labels_file(gold_file_path)
        
        # Check if labels are correct
        if not correct_labels(pred_labels, gold_labels):
            logging.error(f"Label mismatch for {subtask}")
            return None
        
        # Extract matching lists for evaluation
        if subtask in ['1A', '1B']:
            from scorer.task import _extract_matching_lists
            pred_values, gold_values = _extract_matching_lists(pred_labels, gold_labels, subtask)
            acc, precision, recall, f1 = evaluate(pred_labels, gold_labels, subtask)
            
            # Calculate per-class metrics
            class_report = classification_report(gold_values, pred_values, output_dict=True, zero_division=0)
            per_class_results = {}
            for label in class_report:
                if label not in ['accuracy', 'macro avg', 'weighted avg']:
                    per_class_results[label] = {
                        'Precision': class_report[label]['precision'],
                        'Recall': class_report[label]['recall'],
                        'F1 Score': class_report[label]['f1-score'],
                        'Support': class_report[label]['support']
                    }
        else:  # subtask 1C - only hate severity
            from scorer.task import _extract_matching_lists_1C
            pred_values, gold_values = _extract_matching_lists_1C(pred_labels, gold_labels)
            acc, precision, recall, f1 = evaluate_1C(pred_labels, gold_labels)
            
            # Calculate per-class metrics only for hate severity
            per_class_results = {
                'hate_severity': {}
            }
            
            class_report = classification_report(
                gold_values['hate_severity'], 
                pred_values['hate_severity'], 
                output_dict=True, 
                zero_division=0
            )
            for label in class_report:
                if label not in ['accuracy', 'macro avg', 'weighted avg']:
                    per_class_results['hate_severity'][label] = {
                        'Precision': class_report[label]['precision'],
                        'Recall': class_report[label]['recall'],
                        'F1 Score': class_report[label]['f1-score'],
                        'Support': class_report[label]['support']
                    }
        
        return {
            'Subtask': subtask,
            'Accuracy': acc,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'PerClassResults': per_class_results
        }
    except Exception as e:
        logging.error(f"Error calculating metrics for {subtask}: {str(e)}")
        return None

def main():
    """
    Main function to generate combined results table
    """
    # Define file paths
    results = {
        '1A': {
            'pred': 'results/subtask_1A.tsv',
            'gold': 'data/subtask_1A/blp25_hatespeech_subtask_1A_test_with_labels.tsv'
        },
        '1B': {
            'pred': 'results/subtask_1B.tsv',
            'gold': 'data/subtask_1B/blp25_hatespeech_subtask_1B_test_with_labels.tsv'
        },
        '1C': {
            'pred': 'results/subtask_1C.tsv',
            'gold': 'data/subtask_1C/blp25_hatespeech_subtask_1C_test_with_labels.tsv'
        }
    }
    
    # Calculate metrics for all subtasks
    all_results = []
    for subtask, paths in results.items():
        logging.info(f"Processing {subtask}...")
        metrics = calculate_metrics_for_subtask(paths['pred'], paths['gold'], subtask)
        if metrics:
            all_results.append(metrics)
    
    # Also calculate for 1C_severe if it exists
    if os.path.exists('results/subtask_1C_severe.tsv'):
        logging.info("Processing 1C (Severity only)...")
        metrics = calculate_metrics_for_subtask(
            'results/subtask_1C_severe.tsv',
            'data/subtask_1C/blp25_hatespeech_subtask_1C_test_with_labels.tsv',
            '1C_severe'
        )
        if metrics:
            all_results.append(metrics)
    
    # Create DataFrame and display results
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Round to 4 decimal places for better readability
        df = df.round(4)
        
        # Reorder columns for better presentation
        df = df[['Subtask', 'Accuracy', 'Precision', 'Recall', 'F1 Score']]
        
        print("\n" + "="*70)
        print("COMBINED RESULTS TABLE FOR RESEARCH PAPER")
        print("="*70)
        print(df.to_string(index=False))
        print("="*70)
        
        # Save to CSV format
        df.to_csv('combined_results_table.csv', index=False)
        
        # Generate LaTeX table for Overleaf
        generate_latex_table(df)
        
        # Generate per-class results tables
        generate_per_class_tables(all_results)
        
        print("\nResults saved to:")
        print("- combined_results_table.csv")
        print("- combined_results_table.tex")
        print("- per_class_results.tex")
        
        # Also create a markdown version
        markdown_table = df.to_markdown(index=False, tablefmt="grid")
        with open('combined_results_table.md', 'w') as f:
            f.write("# Combined Results Table\n\n")
            f.write(markdown_table)
        print("- combined_results_table.md")
        
    else:
        logging.error("No results were calculated. Please check the input files.")

def generate_latex_table(df):
    """
    Generate LaTeX table in Overleaf format
    """
    latex_content = """\\begin{table}[ht]
  \\centering
  \\begin{tabular}{lcccc}
    \\hline
    \\textbf{Subtask} & \\textbf{Accuracy} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1 Score} \\\\
    \\hline
"""
    
    for _, row in df.iterrows():
        latex_content += f"    {row['Subtask']} & {row['Accuracy']:.4f} & {row['Precision']:.4f} & {row['Recall']:.4f} & {row['F1 Score']:.4f} \\\\\n"
    
    latex_content += """    \\hline
  \\end{tabular}
  \\caption{Overall performance metrics across all subtasks.}
  \\label{tab:overall_results}
\\end{table}
"""
    
    with open('combined_results_table.tex', 'w') as f:
        f.write(latex_content)

def generate_per_class_tables(all_results):
    """
    Generate per-class results tables in LaTeX format
    """
    latex_content = """% Per-class results tables
"""
    
    for result in all_results:
        subtask = result['Subtask']
        per_class = result['PerClassResults']
        
        if subtask in ['1A', '1B']:
            # Find dominant values for each metric
            classes = sorted(per_class.keys())
            precision_values = [per_class[cls]['Precision'] for cls in classes]
            recall_values = [per_class[cls]['Recall'] for cls in classes]
            f1_values = [per_class[cls]['F1 Score'] for cls in classes]
            
            max_precision_idx = np.argmax(precision_values)
            max_recall_idx = np.argmax(recall_values)
            max_f1_idx = np.argmax(f1_values)
            
            latex_content += f"""
\\begin{{table}}[ht]
  \\centering
  \\begin{{tabular}}{{lccc}}
    \\hline
    \\textbf{{Class}} & \\textbf{{Precision}} & \\textbf{{Recall}} & \\textbf{{F1 Score}} \\\\
    \\hline
"""
            
            for i, class_label in enumerate(classes):
                metrics = per_class[class_label]
                precision_str = f"\\textbf{{{metrics['Precision']:.4f}}}" if i == max_precision_idx else f"{metrics['Precision']:.4f}"
                recall_str = f"\\textbf{{{metrics['Recall']:.4f}}}" if i == max_recall_idx else f"{metrics['Recall']:.4f}"
                f1_str = f"\\textbf{{{metrics['F1 Score']:.4f}}}" if i == max_f1_idx else f"{metrics['F1 Score']:.4f}"
                
                latex_content += f"    {class_label} & {precision_str} & {recall_str} & {f1_str} \\\\\n"
            
            latex_content += """    \\hline
  \\end{tabular}
  \\caption{Per-class performance metrics for Subtask """ + subtask + """.}
  \\label{tab:per_class_""" + subtask.lower() + """}
\\end{table}
"""
        else:  # Subtask 1C - only hate severity
            aspect = 'hate_severity'
            aspect_name = 'Hate Severity'
            
            # Find dominant values for each metric
            classes = sorted(per_class[aspect].keys())
            precision_values = [per_class[aspect][cls]['Precision'] for cls in classes]
            recall_values = [per_class[aspect][cls]['Recall'] for cls in classes]
            f1_values = [per_class[aspect][cls]['F1 Score'] for cls in classes]
            
            max_precision_idx = np.argmax(precision_values)
            max_recall_idx = np.argmax(recall_values)
            max_f1_idx = np.argmax(f1_values)
            
            latex_content += f"""
\\begin{{table}}[ht]
  \\centering
  \\begin{{tabular}}{{lccc}}
    \\hline
    \\textbf{{Class}} & \\textbf{{Precision}} & \\textbf{{Recall}} & \\textbf{{F1 Score}} \\\\
    \\hline
"""
            
            for i, class_label in enumerate(classes):
                metrics = per_class[aspect][class_label]
                precision_str = f"\\textbf{{{metrics['Precision']:.4f}}}" if i == max_precision_idx else f"{metrics['Precision']:.4f}"
                recall_str = f"\\textbf{{{metrics['Recall']:.4f}}}" if i == max_recall_idx else f"{metrics['Recall']:.4f}"
                f1_str = f"\\textbf{{{metrics['F1 Score']:.4f}}}" if i == max_f1_idx else f"{metrics['F1 Score']:.4f}"
                
                latex_content += f"    {class_label} & {precision_str} & {recall_str} & {f1_str} \\\\\n"
            
            latex_content += """    \\hline
  \\end{tabular}
  \\caption{Per-class performance metrics for Subtask """ + subtask + """ - """ + aspect_name + """.}
  \\label{tab:per_class_""" + subtask.lower() + """_""" + aspect + """}
\\end{table}
"""
    
    with open('per_class_results.tex', 'w') as f:
        f.write(latex_content)

if __name__ == "__main__":
    main()
