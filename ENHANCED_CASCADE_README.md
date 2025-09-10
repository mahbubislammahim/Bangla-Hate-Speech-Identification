# Enhanced Cascade Implementation for Subtask 1B

## Overview

This implementation combines the cascading approach from `cascade_subtask_1A.py` with the enhanced model architecture from `finetune_subtask_1B.py` to create a more powerful two-stage classification system for hate speech detection in Subtask 1B.

## Architecture

### Stage 1: Binary Classification (None vs Hate)
- **Model**: Standard BanglaBERT with AutoModelForSequenceClassification
- **Purpose**: Quickly filter out "None" (non-hate) samples
- **Classes**: 2 (None=0, Hate=1)
- **Threshold**: Configurable (default: 0.3)

### Stage 2: Enhanced Multiclass Classification
- **Model**: EnhancedBanglaBERT with additional layers
- **Purpose**: Detailed classification of hate speech types
- **Classes**: 5 (None, Society, Organization, Community, Individual)
- **Features**:
  - Attention-based pooling
  - Additional dense layers (768 → 512 → 5)
  - GELU activation
  - Dropout regularization
  - Early stopping

## Key Improvements

1. **Enhanced Model Architecture**: 
   - Uses custom `EnhancedBanglaBERT` class with attention pooling
   - Additional dense layers for better feature learning
   - Improved regularization techniques

2. **Cascading Strategy**:
   - Efficiently routes samples based on binary classification
   - Reduces computational load for obvious "None" cases
   - Focuses enhanced model power on challenging hate speech classification

3. **Training Configuration**:
   - Optimized learning rates and batch sizes
   - Early stopping to prevent overfitting
   - Different training strategies for each stage

## Usage

```python
# Run with default threshold (0.3)
python cascade_subtask_1B_enhanced.py

# Or modify threshold in the script
run_enhanced_cascade(threshold=0.5)
```

## File Structure

```
enhanced_cascade_stage1_1B_binary/     # Stage 1 model outputs
enhanced_cascade_stage2_1B_multiclass/ # Stage 2 model outputs  
enhanced_cascade_outputs/              # Final predictions
├── subtask_1B_dev_enhanced_cascade.tsv
└── subtask_1B_test_enhanced_cascade.tsv
```

## Data Files

The script expects the following data files in the `data/subtask_1B/` directory:
- `blp25_hatespeech_subtask_1B_train.tsv`
- `blp25_hatespeech_subtask_1B_dev.tsv`
- `blp25_hatespeech_subtask_1B_test.tsv`

## Model Performance

The enhanced cascade approach provides:
- Better feature representation through enhanced architecture
- Efficient inference through cascading
- Improved handling of class imbalance
- More robust predictions on challenging samples

## Comparison with Original Approaches

| Feature | cascade_subtask_1A.py | cascade_subtask_1B.py | Enhanced Cascade |
|---------|----------------------|----------------------|------------------|
| Architecture | Standard → Standard | Standard → Standard | Standard → Enhanced |
| Pooling | Standard | Standard | Attention-based |
| Dense Layers | Default | Default | 768→512→classes |
| Early Stopping | No | Yes | Yes |
| Regularization | Basic | Basic | Advanced |

## Configuration Options

Key parameters that can be adjusted:
- `threshold`: Binary classification threshold (default: 0.3)
- `hidden_dropout`: Dropout rate in enhanced model (default: 0.3)
- `use_attention_pooling`: Enable attention pooling (default: True)
- Learning rates, batch sizes, and epochs in training arguments
