import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json
import matplotlib.pyplot as plt
import seaborn as sns


class BugClassifier:
    # creating two classifiers - one for severity and one for bug type
    def __init__(self):
        self.severity_classifier = None
        self.bug_type_classifier = None
        self.vocabulary = None
        self.label_names = {
            'severity': ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'],
            'bug_type': ['TIMING', 'PROTOCOL', 'MEMORY', 'FUNCTIONAL', 'TESTBENCH', 'OTHER']
        }

    def load_data(self):
        # load training data
        with open('features_train.pkl', 'rb') as f:
            train_data = pickle.load(f)
        # load validation data
        with open('features_val.pkl', 'rb') as f:
            val_data = pickle.load(f)
        # load test data
        with open('features_test.pkl', 'rb') as f:
            test_data = pickle.load(f)

        # load vocabulary - vocabulary is the map of words -> index
        with open('vocabulary.pkl', 'rb')as f:
            self.vocabulary = pickle.load(f)

        return train_data, val_data, test_data
    
    def train_severity_classifier(self, X_train, y_train, X_val, y_val):
        # Check the class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        for label_index, count in zip(unique, counts):
            label_name = self.label_names['severity'][label_index]
            print(f" {label_name}: {count} ({count/len(y_train) * 100})")
        
        # Hyperparamter grid
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 20, 30, None],
            "min_sample_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "class_weight": ["balanced", "balanced_subsample"]
        }        

        # Grid search
        base_model = RandomForestClassifier(random_state = 42)

        grid_search = GridSearchCV(
            base_model, 
            param_grid,
            cv=3,
            scoring='f1_weighted',
            n_jobs = 1,
            verbose=True
        )

        grid_search.fit(X_train, y_train)

        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}") # best rounded to 4 decimal points

        self.severity_classifier = grid_search.best_estimator_

        # evaluate on validation set
        y_val_pred = self.severity_classifier.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)

        print(f"Validation accuracy: {val_accuracy:.4f}")
        print("Validation Classification Report")
        print(classification_report(
            y_val,
            y_val_pred,
            target_names=self.label_names['severity'],
            zero_division=False
        ))

        return self.severity_classifier
    
    def train_bug_type_classifier(self, X_train, y_train, X_val, y_val):
        unique, counts = np.unique(y_train, return_counts=True)
        for label_index, count in zip(unique, counts):
            label_name = self.label_names['bug_type'][label_index]
            print(f" {label_name}: {count} ({count}/{len(y_train)*100})")

        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [15, 25, None],
            'min_samples_split': [2, 5],
            'class_weight': ['balanced']
        }

        # Performing grid search
        base_model = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=3,
            scoring='f1_weighted',
            n_jobs=1,
            verbose=True
        )

        grid_search.fit(X_train, y_train)

        # get the best params/estimator
        self.bug_type_classifier = grid_search.best_estimator_

        y_val_pred = self.bug_type_classifier.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)

        print(f"Validation accuracy: {val_accuracy:.4f}")
        print("Validation Classification Report")
        print(classification_report(
            y_val,
            y_val_pred,
            target_names=self.label_names['bug_type'],
            zero_division=False
        ))

        return self.bug_type_classifier
    
    def evaluate_on_test(self, X_test, y_test_severity, y_test_bug_type):
        print("Severity prediction")
        y_severity_pred = self.severity_classifier.predict(X_test)
        severity_accuracy = accuracy_score(y_test_severity, y_severity_pred)

        print(f"Test accuracy: {severity_accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(
            y_test_severity,
            y_severity_pred,
            target_names=self.label_names['severity'],
            zero_division=False
        ))

        self.plot_confusion_matrix(
            y_test_severity,
            y_severity_pred,
            self.label_names['severity'],
            'Severioty Confusion Matrix',
            'severity_confusion_matrix.png'
        )

        #Bug types preidctions
        print("Bug Type prediction")

        y_bug_type_pred = self.bug_type_classifier.predict(X_test)
        bug_type_accuracy = accuracy_score(y_test_bug_type, y_bug_type_pred)

        print(f"Test Accuracy: {bug_type_accuracy:.4f}")
        print("Classification Report")
        print(classification_report(
            y_test_bug_type,
            y_bug_type_pred,
            target_names=self.label_names['bug_type'],
            zero_division=False,
        ))

        self.plot_confusion_matrix(
            y_test_bug_type,
            y_bug_type_pred,
            self.label_names['bug_type'],
            'Bug Type Confusion Matrix',
            'bug_type_confusion_matrix.png'
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
            xticklables=labels,
            yticklabels=labels
        )

        plt.title(title)
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f" Saved confusion matrix to {filename}")
        plt.close()

    
    def save_models(self, filename="bug_classifier.pkl"):
        model_data = {
            'severity_classifier': self.severity_classifier,
            'bug_type_classifier': self.bug_type_classifier,
            'vocabulary': self.vocabulary,
            'label_names': self.label_names
        }

        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Saved models to {filename}")

    # training
    def train(self):
        # load data
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

        print("Data set sizes: ")
        print(f" Train: {len(X_train)} bugs")
        print(f" Validation: {len(X_val)} bugs")
        print(f" Test: {len(X_test)} bugs")

        print(f" Feature dimension: {X_train.shape[1]}")

        self.train_severity_classifier(X_train, y_train_severity, X_val, y_val_severity)
        self.train_bug_type_classifier(X_train, y_train_bug_type, X_val, y_val_bug_type)

        test_results = self.evaluate_on_test(X_test, y_test_severity, y_test_bug_type)

        self.save_models()

        return test_results
    

def main():
    print("Bug Classifier Training")

    classifier = BugClassifier()
    results = classifier.train()

    print("Training Complete!")

    print(f"Final Test Results: ")
    print(f"    Severity Accuracy: {results['severity_accuracy']:.2%}")
    print(f"    Bug Type Accuracy: {results['bug_type_accuracy']:.2%}")
    print("\nFiles created:")
    print("  - bug_classifier.pkl (trained models)")
    print("  - severity_confusion_matrix.png")
    print("  - bug_type_confusion_matrix.png")

if __name__ == "__main__":
    main()




    




        


