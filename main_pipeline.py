"""
HASTIKA @ ICON-2026: Kannada-English Code-Mixed Hate Speech Detection
Novel Ensemble Framework combining MuRIL and XLM-RoBERTa with Focal Loss

Author: Rifah Tasnim Jui
Email: tazfahh@gmail.com
Date: September 2026
"""

import os
import gc
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_PATH = '/content/drive/MyDrive/Hastika_Data/'
MURIL_MODEL_NAME = "google/muril-base-cased"
XLMR_MODEL_NAME = "xlm-roberta-large"
MAX_LENGTH = 128
BATCH_SIZE = 16
MURIL_LR = 2e-5
XLMR_LR = 1e-5
MURIL_EPOCHS = 5
XLMR_EPOCHS = 4
PATIENCE = 2
ENSEMBLE_THRESHOLD = 0.45

# ============================================================================
# GOOGLE DRIVE MOUNT
# ============================================================================
def mount_drive():
    """Mount Google Drive to access data and save models"""
    from google.colab import drive
    drive.mount('/content/drive')
    print("✅ Google Drive mounted successfully")

# ============================================================================
# DATA PREPROCESSING
# ============================================================================
def clean_kanglish(text):
    """
    Clean Kanglish text for better model performance
    - Lowercase conversion
    - URL and mention normalization
    - HTML entity removal
    - Whitespace normalization
    """
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', ' URL ', text)
    text = re.sub(r'@\w+', ' USER ', text)
    text = text.replace('<br>', ' ')
    text = text.replace('&quot;', '"')
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def load_and_preprocess_data():
    """Load training data and perform preprocessing"""
    print("\n📊 Loading and preprocessing data...")
    
    # Load data
    train_bin = pd.read_csv(DATA_PATH + 'binary_train.csv', encoding='utf-8')
    val_inputs = pd.read_csv(DATA_PATH + 'binary_validation_inputs.csv', encoding='utf-8')
    
    # Clean text
    train_bin['Cleaned_Comment'] = train_bin['Comment'].apply(clean_kanglish)
    val_inputs['Cleaned_Comment'] = val_inputs['Comment'].apply(clean_kanglish)
    
    # Split training data for validation
    train_df, val_df = train_test_split(
        train_bin, 
        test_size=0.15, 
        random_state=42, 
        stratify=train_bin['Label']
    )
    
    print(f"✅ Train size: {len(train_df)}, Validation size: {len(val_df)}")
    print(f"✅ Test inputs size: {len(val_inputs)}")
    
    return train_df, val_df, val_inputs

# ============================================================================
# TOKENIZATION
# ============================================================================
def tokenize_data(tokenizer, texts, max_length=MAX_LENGTH):
    """Tokenize texts using the provided tokenizer"""
    return tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )

def prepare_datasets(train_df, val_df, tokenizer):
    """Prepare PyTorch datasets for training"""
    print("\n🔤 Tokenizing data...")
    
    # Label encoding
    label_encoder = LabelEncoder()
    train_df['label_encoded'] = label_encoder.fit_transform(train_df['Label'])
    val_df['label_encoded'] = label_encoder.transform(val_df['Label'])
    
    print(f"Label mapping: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    
    # Tokenize
    train_encodings = tokenize_data(tokenizer, train_df['Cleaned_Comment'].tolist())
    val_encodings = tokenize_data(tokenizer, val_df['Cleaned_Comment'].tolist())
    
    # Create datasets
    train_dataset = TensorDataset(
        train_encodings['input_ids'],
        train_encodings['attention_mask'],
        torch.tensor(train_df['label_encoded'].values)
    )
    
    val_dataset = TensorDataset(
        val_encodings['input_ids'],
        val_encodings['attention_mask'],
        torch.tensor(val_df['label_encoded'].values)
    )
    
    return train_dataset, val_dataset, label_encoder

# ============================================================================
# FOCAL LOSS
# ============================================================================
class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance and focusing on hard examples
    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', device='cpu'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.device = device
        
        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32).to(device)
        else:
            self.alpha = None

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        return torch.sum(focal_loss)

# ============================================================================
# TRAINING PIPELINE
# ============================================================================
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                device, epochs, model_name, patience=PATIENCE):
    """
    Train model with early stopping and validation monitoring
    """
    print(f"\n🚀 Starting {model_name} training...")
    
    best_val_f1 = 0.0
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            total_train_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        val_acc = accuracy_score(all_labels, all_preds)
        
        print(f"\nEpoch {epoch+1}:")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Validation Macro-F1: {val_f1:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")
        print(classification_report(all_labels, all_preds, target_names=['Hate', 'Non-Hate']))
        print("-" * 50)
        
        # Early stopping check
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print(f"✅ New Best Model Saved! (F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            print(f"⚠️ No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("🛑 Early Stopping triggered!")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)
        print(f"\n🏆 Final Best Validation Macro-F1: {best_val_f1:.4f}")
    
    return model, best_val_f1

def train_muril(train_dataset, val_dataset, device):
    """Train MuRIL model with Focal Loss"""
    print("\n" + "="*70)
    print("TRAINING MuRIL MODEL")
    print("="*70)
    
    # Clear memory
    gc.collect()
    torch.cuda.empty_cache()
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        MURIL_MODEL_NAME, 
        num_labels=2
    ).to(device)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Compute class weights
    labels = train_dataset.tensors[2].numpy()
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    alpha_weights = weights / np.sum(weights)
    print(f"Focal Loss Alpha Weights: {alpha_weights}")
    
    # Setup loss, optimizer, scheduler
    criterion = FocalLoss(alpha=alpha_weights, gamma=2.0, reduction='mean', device=device)
    optimizer = AdamW(model.parameters(), lr=MURIL_LR, weight_decay=0.01, eps=1e-8)
    total_steps = len(train_loader) * MURIL_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_steps * 0.1), 
        num_training_steps=total_steps
    )
    
    # Train
    model, best_f1 = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, MURIL_EPOCHS, "MuRIL"
    )
    
    # Save model
    save_path = DATA_PATH + 'best_muril_focal_model.bin'
    torch.save(model.state_dict(), save_path)
    print(f"✅ MuRIL model saved to {save_path}")
    
    return model

def train_xlmr(train_dataset, val_dataset, device):
    """Train XLM-RoBERTa model with Focal Loss"""
    print("\n" + "="*70)
    print("TRAINING XLM-RoBERTa MODEL")
    print("="*70)
    
    # Clear memory
    gc.collect()
    torch.cuda.empty_cache()
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        XLMR_MODEL_NAME, 
        num_labels=2
    ).to(device)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Compute class weights
    labels = train_dataset.tensors[2].numpy()
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    alpha_weights = weights / np.sum(weights)
    print(f"Focal Loss Alpha Weights: {alpha_weights}")
    
    # Setup loss, optimizer, scheduler
    criterion = FocalLoss(alpha=alpha_weights, gamma=2.0, reduction='mean', device=device)
    optimizer = AdamW(model.parameters(), lr=XLMR_LR, weight_decay=0.01, eps=1e-8)
    total_steps = len(train_loader) * XLMR_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_steps * 0.1), 
        num_training_steps=total_steps
    )
    
    # Train
    model, best_f1 = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, XLMR_EPOCHS, "XLM-RoBERTa"
    )
    
    # Save model
    save_path = DATA_PATH + 'best_xlmr_model.bin'
    torch.save(model.state_dict(), save_path)
    print(f"✅ XLM-R model saved to {save_path}")
    
    return model

# ============================================================================
# ENSEMBLE INFERENCE
# ============================================================================
def load_trained_models(device):
    """Load trained MuRIL and XLM-RoBERTa models"""
    print("\n🔄 Loading trained models...")
    
    # Load MuRIL
    muril_tokenizer = AutoTokenizer.from_pretrained(MURIL_MODEL_NAME)
    muril_model = AutoModelForSequenceClassification.from_pretrained(
        MURIL_MODEL_NAME, 
        num_labels=2
    )
    muril_model.load_state_dict(
        torch.load(DATA_PATH + 'best_muril_focal_model.bin', map_location=device)
    )
    muril_model.to(device)
    muril_model.eval()
    
    # Load XLM-RoBERTa
    xlmr_tokenizer = AutoTokenizer.from_pretrained(XLMR_MODEL_NAME)
    xlmr_model = AutoModelForSequenceClassification.from_pretrained(
        XLMR_MODEL_NAME, 
        num_labels=2
    )
    xlmr_model.load_state_dict(
        torch.load(DATA_PATH + 'best_xlmr_model.bin', map_location=device)
    )
    xlmr_model.to(device)
    xlmr_model.eval()
    
    print("✅ Models loaded successfully")
    return muril_model, muril_tokenizer, xlmr_model, xlmr_tokenizer

def generate_ensemble_predictions(val_inputs, muril_model, muril_tokenizer, 
                                  xlmr_model, xlmr_tokenizer, device):
    """Generate ensemble predictions from both models"""
    print("\n🎯 Generating ensemble predictions...")
    
    # Clean test data
    val_inputs['Cleaned_Comment'] = val_inputs['Comment'].apply(clean_kanglish)
    texts = val_inputs['Cleaned_Comment'].tolist()
    
    # Tokenize for both models
    muril_encodings = tokenize_data(muril_tokenizer, texts)
    xlmr_encodings = tokenize_data(xlmr_tokenizer, texts)
    
    # Generate predictions
    muril_probs_list = []
    xlmr_probs_list = []
    batch_size = 32
    
    # MuRIL predictions
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_ids = muril_encodings['input_ids'][i:i+batch_size].to(device)
            batch_mask = muril_encodings['attention_mask'][i:i+batch_size].to(device)
            outputs = muril_model(input_ids=batch_ids, attention_mask=batch_mask)
            probs = F.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
            muril_probs_list.extend(probs)
    
    # XLM-R predictions
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_ids = xlmr_encodings['input_ids'][i:i+batch_size].to(device)
            batch_mask = xlmr_encodings['attention_mask'][i:i+batch_size].to(device)
            outputs = xlmr_model(input_ids=batch_ids, attention_mask=batch_mask)
            probs = F.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
            xlmr_probs_list.extend(probs)
    
    # Ensemble (probability averaging)
    muril_probs = np.array(muril_probs_list)
    xlmr_probs = np.array(xlmr_probs_list)
    ensemble_probs = (muril_probs + xlmr_probs) / 2.0
    
    # Apply threshold
    predictions = (ensemble_probs >= ENSEMBLE_THRESHOLD).astype(int)
    
    # Map to labels
    label_mapping = {0: 'Hate', 1: 'Non-Hate'}
    val_inputs['label'] = [label_mapping[p] for p in predictions]
    
    return val_inputs

def save_submission(val_inputs):
    """Save final submission file"""
    submission = val_inputs[['id', 'label']].copy()
    submission_path = DATA_PATH + 'predictions_ensemble_taskA.csv'
    submission.to_csv(submission_path, index=False)
    
    print(f"\n✅ Ensemble predictions saved to: {submission_path}")
    print("\n--- Sample Submission (First 5 rows) ---")
    print(submission.head())
    
    return submission_path

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Main execution pipeline"""
    print("="*70)
    print("HASTIKA @ ICON-2026: KANGLISH HATE SPEECH DETECTION")
    print("Novel Ensemble Framework: MuRIL + XLM-RoBERTa with Focal Loss")
    print("="*70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Mount Drive
    mount_drive()
    
    # Load and preprocess data
    train_df, val_df, val_inputs = load_and_preprocess_data()
    
    # Initialize tokenizer (using MuRIL for dataset preparation)
    tokenizer = AutoTokenizer.from_pretrained(MURIL_MODEL_NAME)
    
    # Prepare datasets
    train_dataset, val_dataset, label_encoder = prepare_datasets(
        train_df, val_df, tokenizer
    )
    
    # Train MuRIL
    muril_model = train_muril(train_dataset, val_dataset, device)
    
    # Train XLM-RoBERTa
    xlmr_model = train_xlmr(train_dataset, val_dataset, device)
    
    # Load trained models for inference
    muril_model, muril_tokenizer, xlmr_model, xlmr_tokenizer = load_trained_models(device)
    
    # Generate ensemble predictions
    val_inputs = generate_ensemble_predictions(
        val_inputs, muril_model, muril_tokenizer, 
        xlmr_model, xlmr_tokenizer, device
    )
    
    # Save submission
    submission_path = save_submission(val_inputs)
    
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"Final submission file: {submission_path}")
    print("\n📊 Next Steps:")
    print("1. Download the submission file from Google Drive")
    print("2. Upload to Codalab competition page")
    print("3. Check leaderboard for Macro-F1 score")

if __name__ == "__main__":
    main()
