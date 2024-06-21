import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_time_series(data, title='Time Series Data', xlabel='Time', ylabel='Value'):
    """
    Plota uma série temporal.

    Args:
        data (pd.Series or np.ndarray): Dados da série temporal.
        title (str): Título do gráfico.
        xlabel (str): Rótulo do eixo X.
        ylabel (str): Rótulo do eixo Y.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def plot_training_history(history):
    """
    Plota a acurácia e perda do treinamento do modelo.

    Args:
        history (dict): Histórico de treinamento do modelo, contendo as chaves 'accuracy', 'loss', 'val_accuracy', 'val_loss'.
    """
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))

    # Plotando a acurácia
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plotando a perda
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


def plot_confusion_matrix(conf_matrix, class_names):
    """
    Plota a matriz de confusão.

    Args:
        conf_matrix (np.ndarray): Matriz de confusão.
        class_names (list): Lista de nomes das classes.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


def plot_feature_importance(feature_importance, feature_names):
    """
    Plota a importância das features.

    Args:
        feature_importance (np.ndarray): Importância das features.
        feature_names (list): Lista de nomes das features.
    """
    sorted_idx = np.argsort(feature_importance)
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance')
    plt.show()

