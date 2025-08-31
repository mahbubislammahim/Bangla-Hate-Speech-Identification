# ============================================================================
# Load and Use Saved BanglaBERT Model - OFFICIAL SCORER FORMAT
# ============================================================================

import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ============================================================================
# CELL 1: Load Your Saved Model from Hugging Face
# ============================================================================

# Replace with your actual Hugging Face repository name
MODEL_NAME = "Mahim47/banglabert-hatespeech-subtask1b"  # Your saved model
# MODEL_NAME = "Mahim47/banglabert-hatespeech-subtask1b-v2"  # Or use version 2 if you have it

print(f"🤖 Loading model from: {MODEL_NAME}")

# Load the saved model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Set model to evaluation mode
model.eval()

print("✅ Model loaded successfully!")

# ============================================================================
# CELL 2: Define Label Mapping
# ============================================================================

# Label mapping (same as training)
l2id = {'None': 0, 'Society': 1, 'Organization': 2, 'Community': 3, 'Individual': 4}
id2l = {v: k for k, v in l2id.items()}

print(f"📋 Label mapping: {l2id}")

# ============================================================================
# CELL 3: Prediction Function
# ============================================================================

def predict_hate_speech(text, max_length=128):
    """
    Predict hate speech target for a given Bengali text.
    
    Args:
        text (str): Bengali text to classify
        max_length (int): Maximum sequence length
    
    Returns:
        dict: Prediction results with label and confidence
    """
    # Tokenize the input text
    inputs = tokenizer(
        text,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt"
    )
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # Get label name
    predicted_label = id2l[predicted_class]
    
    return {
        "text": text,
        "predicted_label": predicted_label,
        "confidence": confidence,
        "all_probabilities": {
            id2l[i]: prob.item() for i, prob in enumerate(probabilities[0])
        }
    }

# ============================================================================
# CELL 4: Load Test Data and Generate Predictions
# ============================================================================

print("\n" + "="*60)
print("📊 GENERATING PREDICTIONS ON TEST DATA")
print("="*60)

# Load test data
test_file = 'blp25_hatespeech_subtask_1B_dev_test.tsv'
test_df = pd.read_csv(test_file, sep='\t')

print(f"📁 Loaded {len(test_df)} test samples")

# Generate predictions
predictions = []
for idx, row in test_df.iterrows():
    text = row['text']  # Assuming 'text' is the column name
    result = predict_hate_speech(text)
    predictions.append({
        'id': row['id'],
        'text': text,
        'predicted_label': result['predicted_label'],
        'confidence': result['confidence']
    })

# Create results DataFrame
results_df = pd.DataFrame(predictions)
print(f"✅ Generated predictions for {len(results_df)} samples")

# ============================================================================
# CELL 5: Save Predictions in OFFICIAL SCORER FORMAT
# ============================================================================

# Official format: id \t label \t model
output_file = f"subtask_1B_banglabert_official.tsv"

with open(output_file, "w", encoding='utf-8') as f:
    f.write("id\tlabel\tmodel\n")  # Header as required by scorer
    for _, row in results_df.iterrows():
        f.write(f"{row['id']}\t{row['predicted_label']}\t{MODEL_NAME}\n")

print(f"\n💾 Predictions saved in OFFICIAL FORMAT: {output_file}")

# ============================================================================
# CELL 6: Official Scorer Evaluation (if validation data available)
# ============================================================================

# Check if validation data is available for evaluation
validation_file = 'blp25_hatespeech_subtask_1B_dev.tsv'

try:
    print(f"\n🏆 OFFICIAL SCORER EVALUATION:")
    print("=" * 50)
    
    # Load validation data
    val_df = pd.read_csv(validation_file, sep='\t')
    print(f"📁 Loaded {len(val_df)} validation samples")
    
    # Generate predictions on validation data
    val_predictions = {}
    val_gold_labels = {}
    
    for idx, row in val_df.iterrows():
        doc_id = str(row['id'])
        text = row['text']
        gold_label = str(row['label'])
        
        # Get prediction
        result = predict_hate_speech(text)
        predicted_label = result['predicted_label']
        
        # Store for evaluation
        val_predictions[doc_id] = predicted_label
        val_gold_labels[doc_id] = gold_label
    
    # Run EXACT official evaluation (same as competition)
    def _extract_matching_lists(pred_labels, gold_labels, subtask):
        """EXACT copy from scorer/task.py"""
        pred_values, gold_values = ([], [])
        for k in gold_labels.keys():
            pred_values.append(pred_labels[k])
            gold_values.append(gold_labels[k])
        return pred_values, gold_values
    
    def evaluate_official(pred_labels, gold_labels, subtask):
        """EXACT copy from scorer/task.py"""
        pred_values, gold_values = _extract_matching_lists(pred_labels, gold_labels, subtask)
        acc = accuracy_score(gold_values, pred_values)
        precision = precision_score(gold_values, pred_values, average='weighted')
        recall = recall_score(gold_values, pred_values, average='weighted')
        f1 = f1_score(gold_values, pred_values, average='micro')
        return acc, precision, recall, f1
    
    # Run evaluation
    acc, precision, recall, f1 = evaluate_official(val_predictions, val_gold_labels, '1B')
    
    print(f"📊 OFFICIAL COMPETITION SCORES:")
    print(f"   Accuracy: {acc:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1: {f1:.4f}")
    
    print("\n🎯 These are the EXACT metrics the competition will use!")
    print("✅ Using identical functions from scorer/task.py")
    
except FileNotFoundError:
    print(f"\n⚠️  Validation file '{validation_file}' not found. Skipping evaluation.")
    print("📊 You can run evaluation later when validation data is available.")

# ============================================================================
# CELL 7: Summary Statistics
# ============================================================================

print("\n📈 PREDICTION SUMMARY:")
print("="*40)

# Count predictions by label
label_counts = results_df['predicted_label'].value_counts()
print("Label distribution:")
for label, count in label_counts.items():
    percentage = (count / len(results_df)) * 100
    print(f"   {label}: {count} ({percentage:.1f}%)")

# Average confidence
avg_confidence = results_df['confidence'].mean()
print(f"\n🎯 Average confidence: {avg_confidence:.4f}")

print(f"\n🎉 All done! Model loaded and predictions generated!")
print(f"📁 Results saved in: {output_file}")
print(f"📋 File format: id\\tlabel\\tmodel (Official scorer format)")
print(f"🚀 Ready for submission to competition!")
