# train_classifier.py

import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from paths import FEATURES_TRAIN, FEATURES_VAL, FEATURES_TEST, VOCABULARY, BUG_CLASSIFIER, MODELS_DIR
import json

class BugClassifier:
    
    def __init__(self):
        self.severity_classifier = None
        self.bug_type_classifier = None
        self.vocabulary = None
        self.label_names = {
            'severity': ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'],
            'bug_type': ['TIMING', 'PROTOCOL', 'MEMORY', 'FUNCTIONAL', 'TESTBENCH', 'OTHER']
        }
    
    def load_data(self):
        print("Loading training data...")
        with open(FEATURES_TRAIN, 'rb') as f:
            train_data = pickle.load(f)
        
        print("Loading validation data...")
        with open(FEATURES_VAL, 'rb') as f:
            val_data = pickle.load(f)
        
        print("Loading test data...")
        with open(FEATURES_TEST, 'rb') as f:
            test_data = pickle.load(f)
        
        print("Loading vocabulary...")
        with open(VOCABULARY, 'rb') as f:
            self.vocabulary = pickle.load(f)
        
        return train_data, val_data, test_data
    
    def train_severity_classifier(self, X_train, y_train, X_val, y_val):
        print("\n" + "="*60)
        print("TRAINING SEVERITY CLASSIFIER")
        print("="*60)
        
        print("\nClass distribution (training):")
        unique, counts = np.unique(y_train, return_counts=True)
        for label_idx, count in zip(unique, counts):
            label_name = self.label_names['severity'][label_idx]
            print(f"  {label_name}: {count} ({count/len(y_train)*100:.1f}%)")
        
        # Hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced', 'balanced_subsample']
        }
        
        print("\nPerforming grid search...")
        base_model = RandomForestClassifier(random_state=42)
        
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=3,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        self.severity_classifier = grid_search.best_estimator_
        
        # Evaluate on validation set
        y_val_pred = self.severity_classifier.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        
        print(f"\nValidation accuracy: {val_accuracy:.4f}")
        print("\nValidation Classification Report:")
        print(classification_report(
            y_val, 
            y_val_pred,
            target_names=self.label_names['severity'],
            zero_division=0
        ))
        
        return self.severity_classifier
    
    def train_bug_type_classifier(self, X_train, y_train, X_val, y_val):
        print("\n" + "="*60)
        print("TRAINING BUG TYPE CLASSIFIER")
        print("="*60)
        
        print("\nClass distribution (training):")
        unique, counts = np.unique(y_train, return_counts=True)
        for label_idx, count in zip(unique, counts):
            label_name = self.label_names['bug_type'][label_idx]
            print(f"  {label_name}: {count} ({count/len(y_train)*100:.1f}%)")
        
        # Simpler grid for bug type (more classes, might need different params)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [15, 25, None],
            'min_samples_split': [2, 5, 10],
            'class_weight': ['balanced']
        }
        
        print("\nPerforming grid search...")
        base_model = RandomForestClassifier(random_state=42)
        
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=3,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        self.bug_type_classifier = grid_search.best_estimator_
        
        # Evaluate on validation set
        y_val_pred = self.bug_type_classifier.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        
        print(f"\nValidation accuracy: {val_accuracy:.4f}")
        print("\nValidation Classification Report:")
        print(classification_report(
            y_val, 
            y_val_pred,
            target_names=self.label_names['bug_type'],
            zero_division=0
        ))
        
        return self.bug_type_classifier
    
    def evaluate_on_test(self, X_test, y_test_severity, y_test_bug_type):
        print("\n" + "="*60)
        print("FINAL TEST SET EVALUATION")
        print("="*60)
        
        print("\n--- SEVERITY PREDICTION ---")
        y_severity_pred = self.severity_classifier.predict(X_test)
        severity_accuracy = accuracy_score(y_test_severity, y_severity_pred)
        
        print(f"Test Accuracy: {severity_accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(
            y_test_severity,
            y_severity_pred,
            target_names=self.label_names['severity'],
            zero_division=0
        ))
        
        # Confusion matrix
        self.plot_confusion_matrix(
            y_test_severity,
            y_severity_pred,
            self.label_names['severity'],
            'Severity Confusion Matrix',
            str(MODELS_DIR / 'severity_confusion_matrix.png')
        )
        
        # Bug Type
        print("\n--- BUG TYPE PREDICTION ---")
        y_bug_type_pred = self.bug_type_classifier.predict(X_test)
        bug_type_accuracy = accuracy_score(y_test_bug_type, y_bug_type_pred)
        
        print(f"Test Accuracy: {bug_type_accuracy:.4f}")
        print(len(self.label_names['bug_type']))
        print(self.label_names['bug_type'])
        print(y_test_bug_type)
        print("\nClassification Report:")
        print(classification_report(
            y_true=y_test_bug_type,
            y_pred=y_bug_type_pred,
            labels=[0, 1, 2, 3, 4, 5],
            target_names=self.label_names['bug_type'],
            zero_division=0
        ))
        
        self.plot_confusion_matrix(
            y_test_bug_type,
            y_bug_type_pred,
            self.label_names['bug_type'],
            'Bug Type Confusion Matrix',
            str(MODELS_DIR / 'bug_type_confusion_matrix.png')
        )
        
        return {
            'severity_accuracy': severity_accuracy,
            'bug_type_accuracy': bug_type_accuracy
        }
    
    def plot_confusion_matrix(self, y_true, y_pred, labels, title, filename):
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels
        )
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  Saved confusion matrix to {filename}")
        plt.close()
    
    def analyze_feature_importance(self):
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        print("\n--- Top 10 Features for Severity ---")
        severity_importance = self.severity_classifier.feature_importances_
        top_10_indices = np.argsort(severity_importance)[-10:][::-1]
        
        for i, idx in enumerate(top_10_indices, 1):
            if idx < 384:  # Embedding features
                print(f"{i}. Embedding dimension {idx}: {severity_importance[idx]:.4f}")
            else:
                print(f"{i}. Structured feature {idx-384}: {severity_importance[idx]:.4f}")
        
        print("\n--- Top 10 Features for Bug Type ---")
        bug_type_importance = self.bug_type_classifier.feature_importances_
        top_10_indices = np.argsort(bug_type_importance)[-10:][::-1]
        
        for i, idx in enumerate(top_10_indices, 1):
            if idx < 384:
                print(f"{i}. Embedding dimension {idx}: {bug_type_importance[idx]:.4f}")
            else:
                print(f"{i}. Structured feature {idx-384}: {bug_type_importance[idx]:.4f}")
    
    def save_models(self, filename='bug_classifier.pkl'):
        model_data = {
            'severity_classifier': self.severity_classifier,
            'bug_type_classifier': self.bug_type_classifier,
            'vocabulary': self.vocabulary,
            'label_names': self.label_names
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\nSaved models to {filename}")
    
    def train(self):
        train_data, val_data, test_data = self.load_data()
        
        X_train = train_data['X']
        y_train_severity = train_data['y_severity']
        y_train_bug_type = train_data['y_bug_type']
        
        X_val = val_data['X']
        y_val_severity = val_data['y_severity']
        y_val_bug_type = val_data['y_bug_type']
        
        X_test = test_data['X']
        y_test_severity = test_data['y_severity']
        y_test_bug_type = test_data['y_bug_type']
        
        print(f"\nDataset sizes:")
        print(f"  Train: {len(X_train)} bugs")
        print(f"  Val: {len(X_val)} bugs")
        print(f"  Test: {len(X_test)} bugs")
        print(f"  Feature dimension: {X_train.shape[1]}")
        
        # Train both classifiers
        self.train_severity_classifier(X_train, y_train_severity, X_val, y_val_severity)
        self.train_bug_type_classifier(X_train, y_train_bug_type, X_val, y_val_bug_type)
        
        # Evaluate on test set
        test_results = self.evaluate_on_test(X_test, y_test_severity, y_test_bug_type)
        
        # Feature importance
        self.analyze_feature_importance()
        
        # Save models
        self.save_models()
        
        return test_results

# Main training script
def main():
    print("="*60)
    print("BUG CLASSIFIER TRAINING PIPELINE")
    print("="*60)
    
    classifier = BugClassifier()
    results = classifier.train()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nFinal Test Results:")
    print(f"  Severity Accuracy: {results['severity_accuracy']:.2%}")
    print(f"  Bug Type Accuracy: {results['bug_type_accuracy']:.2%}")
    print("\nFiles created:")
    print("  - models/bug_classifier.pkl (trained models)")
    print("  - models/severity_confusion_matrix.png")
    print("  - models/bug_type_confusion_matrix.png")

if __name__ == "__main__":
    main()