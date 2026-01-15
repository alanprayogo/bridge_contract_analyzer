import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import os
import joblib
import logging
import seaborn as sns
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_random_forest(X_train, X_test, y_suit_train, y_suit_test, y_category_train, y_category_test, saved_dir):
    logger.info("Starting Random Forest training")

    if X_train.empty or X_test.empty:
        logger.error("Empty training or testing data")
        raise ValueError("Empty training or testing data")
    if len(y_suit_train) == 0 or len(y_category_train) == 0:
        logger.error("Empty target arrays")
        raise ValueError("Empty target arrays")

    logger.info(f"Training set size: {len(X_train)} samples")
    logger.info(f"Test set size: {len(X_test)} samples")
    logger.info(f"Suit distribution in train: {np.bincount(y_suit_train)}")
    logger.info(f"Suit distribution in test: {np.bincount(y_suit_test)}")
    logger.info(f"Category distribution in train: {np.bincount(y_category_train)}")
    logger.info(f"Category distribution in test: {np.bincount(y_category_test)}")
    logger.info(f"Unique suit classes in y_suit_train: {np.unique(y_suit_train)}")
    logger.info(f"Unique category classes in y_category_train: {np.unique(y_category_train)}")

    logger.info("Training Random Forest for suit...")
    rf_suit = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf_suit.fit(X_train, y_suit_train)

    logger.info("Training Random Forest for category...")
    rf_category = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf_category.fit(X_train, y_category_train)

    logger.info("Evaluating models...")
    suit_pred = rf_suit.predict(X_test)
    category_pred = rf_category.predict(X_test)

    logger.info(f"Unique suit classes in suit_pred: {np.unique(suit_pred)}")
    logger.info(f"Unique category classes in category_pred: {np.unique(category_pred)}")

    try:
        suit_metrics = precision_recall_fscore_support(y_suit_test, suit_pred, average='macro', zero_division=0)
        category_metrics = precision_recall_fscore_support(y_category_test, category_pred, average='macro', zero_division=0)
        logger.info(f"Suit Metrics (Precision, Recall, F1, Support): {suit_metrics}")
        logger.info(f"Category Metrics (Precision, Recall, F1, Support): {category_metrics}")
        logger.info(f"Suit Confusion Matrix:\n{confusion_matrix(y_suit_test, suit_pred)}")
        logger.info(f"Category Confusion Matrix:\n{confusion_matrix(y_category_test, category_pred)}")
    except ValueError as e:
        logger.error(f"Evaluation error: {e}")
        raise

    ss_accuracy = np.mean(suit_pred == y_suit_test)
    sc_accuracy = np.mean(category_pred == y_category_test)
    cp_accuracy = np.mean((suit_pred == y_suit_test) & (category_pred == y_category_test))
    logger.info(f"SS Accuracy: {ss_accuracy:.3f}, SC Accuracy: {sc_accuracy:.3f}, CP Accuracy: {cp_accuracy:.3f}")

    try:
        os.makedirs(saved_dir, exist_ok=True)
        joblib.dump(rf_suit, os.path.join(saved_dir, 'rf_suit.pkl'))
        joblib.dump(rf_category, os.path.join(saved_dir, 'rf_category.pkl'))
        logger.info(f"Models saved to {saved_dir}")
    except PermissionError:
        logger.error(f"Cannot write to directory {saved_dir}")
        raise

    return rf_suit, rf_category, suit_pred, category_pred, ss_accuracy, sc_accuracy, cp_accuracy

def plot_conf_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {title}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, feature_names, title):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=np.array(feature_names)[indices])
    plt.title(f'Feature Importance - {title}')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

def plot_class_distribution(y, title):
    unique, counts = np.unique(y, return_counts=True)
    plt.figure(figsize=(6, 4))
    sns.barplot(x=unique, y=counts)
    plt.title(f'Class Distribution - {title}')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

def plot_regression_result(y_true, y_pred, title=''):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.7, color='royalblue')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{title}: Predicted vs Actual')
    plt.grid(True)
    plt.show()

def plot_accuracies(ss_accuracy, sc_accuracy, cp_accuracy):
    accuracies = [ss_accuracy, sc_accuracy, cp_accuracy]
    labels = ['Suit Accuracy (SS)', 'Category Accuracy (SC)', 'Combined Accuracy (CP)']
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=labels, y=accuracies, palette='viridis')
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.title('Random Forest Prediction Accuracies')
    plt.grid(axis='y')
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.02, f"{acc:.2f}", ha='center')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(base_dir, '../data/processed')
    saved_dir = os.path.join(base_dir, 'saved')

    try:
        logger.info(f"Loading preprocessed data from {processed_dir}")
        X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv'))
        X_test = pd.read_csv(os.path.join(processed_dir, 'X_test.csv'))
        y_suit_train = np.load(os.path.join(processed_dir, 'y_suit_train.npy'))
        y_suit_test = np.load(os.path.join(processed_dir, 'y_suit_test.npy'))
        y_category_train = np.load(os.path.join(processed_dir, 'y_category_train.npy'))
        y_category_test = np.load(os.path.join(processed_dir, 'y_category_test.npy'))

        rf_suit, rf_category, suit_pred, category_pred, ss_accuracy, sc_accuracy, cp_accuracy = train_random_forest(
            X_train, X_test, y_suit_train, y_suit_test, y_category_train, y_category_test, saved_dir
        )

        # Visualisasi
        plot_conf_matrix(y_suit_test, suit_pred, 'Suit')
        plot_conf_matrix(y_category_test, category_pred, 'Category')

        plot_feature_importance(rf_suit, X_train.columns, 'Suit')
        plot_feature_importance(rf_category, X_train.columns, 'Category')

        plot_class_distribution(y_suit_train, 'Suit Train')
        plot_class_distribution(y_category_train, 'Category Train')

        plot_accuracies(ss_accuracy, sc_accuracy, cp_accuracy)

    except FileNotFoundError as e:
        logger.error(f"Preprocessed data not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
