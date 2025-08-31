# Task C: Multi-Label Hate Speech Classification

Excellent work on Task B! Now for Task C - this is a more complex **multi-label classification** task that predicts **three labels simultaneously**:

## 🎯 **Task C Overview:**

Unlike Task B (single classification), Task C predicts **3 labels at once**:
1. **`hate_type`**: None, Abusive, Political Hate
2. **`hate_severity`**: Little to None, Mild, Severe  
3. **`to_whom`**: None, Individual, Organization, Community, Society

## 📊 **Data Structure:**

**Training/Validation:** `id | text | hate_type | hate_severity | to_whom`
**Test:** `id | text` (predict all 3 labels)

## 🚀 **How to Use:**

### **Step 1: Copy the Script**
- Use `subtask_1C_BanglaBERT_Large_Colab.py`
- Split into **21 cells** at `# ============================================================================` markers

### **Step 2: Upload Data Files**
- `blp25_hatespeech_subtask_1C_train.tsv`
- `blp25_hatespeech_subtask_1C_dev.tsv`
- `blp25_hatespeech_subtask_1C_dev_test.tsv`

### **Step 3: Run in Google Colab**
- Enable GPU (required for 335M parameter model)
- Run cells 1-21 sequentially
- Training time: ~45-90 minutes

## 🔧 **Key Technical Features:**

### **1. Custom Multi-Label Model**
```python
class BanglaBERTForMultiLabelClassification(BertPreTrainedModel):
    # Separate classifiers for each task
    self.hate_type_classifier = nn.Linear(config.hidden_size, num_hate_type_labels)
    self.hate_severity_classifier = nn.Linear(config.hidden_size, num_hate_severity_labels)
    self.to_whom_classifier = nn.Linear(config.hidden_size, num_to_whom_labels)
```

### **2. Multi-Label Loss Function**
- Combines losses from all 3 tasks
- `total_loss = hate_type_loss + hate_severity_loss + to_whom_loss`

### **3. Comprehensive Metrics**
- **Overall Accuracy**: All 3 predictions correct
- **Individual Task Accuracy**: Per-task performance
- **F1 Scores**: Macro-averaged for each task
- **Average F1**: Combined performance metric

## 📈 **Expected Performance:**

| Metric | Expected Range |
|--------|----------------|
| **Overall Accuracy** | 60-70% (all 3 correct) |
| **Hate Type Accuracy** | 75-85% |
| **Hate Severity Accuracy** | 70-80% |
| **To Whom Accuracy** | 65-75% |
| **Average F1** | 70-80% |

## 🎯 **Why This Approach Works:**

### **1. Shared Representation**
- Single BanglaBERT encoder for all tasks
- Learns common features across all labels

### **2. Task-Specific Heads**
- Separate classifiers prevent interference
- Each task optimized independently

### **3. Joint Training**
- All tasks learned simultaneously
- Better feature learning through multi-task setup

## 📁 **Output Format:**

The model generates: `subtask_1C_banglabert_large.tsv`
```
id    hate_type    hate_severity    to_whom    model
123   Abusive      Mild            Individual  csebuetnlp/banglabert_large
456   None         Little to None  None        csebuetnlp/banglabert_large
```

## 🔍 **Key Differences from Task B:**

| Aspect | Task B | Task C |
|--------|--------|--------|
| **Labels** | 1 label (to_whom) | 3 labels (hate_type + hate_severity + to_whom) |
| **Model** | Standard classifier | Custom multi-label model |
| **Loss** | Single CrossEntropy | Combined loss from 3 tasks |
| **Metrics** | Simple accuracy | Multiple accuracy + F1 scores |
| **Complexity** | Single classification | Multi-label classification |
| **Training Time** | ~30-45 min | ~45-90 min |

## ⚠️ **Important Notes:**

1. **More Complex**: Multi-label is harder than single-label
2. **Custom Architecture**: Uses custom model class
3. **Memory Usage**: Slightly higher due to multiple heads
4. **Evaluation**: Multiple metrics to track
5. **Output Format**: Different TSV format with 3 prediction columns

## 🏆 **Success Tips:**

1. **Monitor All Metrics**: Don't just look at overall accuracy
2. **Check Label Distribution**: Ensure balanced predictions
3. **Validate Output Format**: Must match expected TSV structure
4. **GPU Required**: 335M parameter model needs GPU
5. **Patience**: Multi-label training takes longer

This is a more advanced task, but the script handles all the complexity for you! 🚀

