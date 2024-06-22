import numpy as np
from src.data.data_loader import download_data
from src.data.data_preprocessing import preprocess_data, create_dataset
from src.visualize.visualize import plot_time_series, plot_training_history, plot_confusion_matrix, \
    plot_feature_importance
from src.models.train import train_model
from utils.config import load_config


def main() -> None:
    config = load_config()

    # Acessar as configurações
    ticker = config['ticker']
    start_date = config['start_date']
    end_date = config['end_date']
    time_step = config['time_step']
    epochs = config['epochs']
    batch_size = config['batch_size']

    # Carregar e processar dados
    data = download_data(ticker, start_date, end_date)
    scaled_data, scaler = preprocess_data(data)
    X, y = create_dataset(scaled_data, time_step) # Rede Neural LSTM

    # Dividir dados em treinamento e teste
    train_size = int(len(X) * 0.8)
    test_size = len(X) - train_size
    X_train, X_test = X[0:train_size], X[train_size:len(X)]
    y_train, y_test = y[0:train_size], y[train_size:len(y)]

    # Treinar o modelo
    model, history = train_model(X_train, y_train, X_test, y_test)

    # Salvar modelo e histórico, gerar visualizações, etc.

    # Plotar a série temporal dos dados brutos
    plot_time_series(data['close'], title='Stock Prices Over Time', xlabel='Date', ylabel='Price')

    # Supondo que você já treinou um modelo e obteve o histórico do treinamento
    history = {
        'accuracy': [0.8, 0.85, 0.87, 0.9],
        'val_accuracy': [0.75, 0.8, 0.82, 0.85],
        'loss': [0.4, 0.35, 0.3, 0.25],
        'val_loss': [0.45, 0.4, 0.38, 0.35]
    }

    # Plotar o histórico do treinamento
    plot_training_history(history)

    # Supondo que você tenha uma matriz de confusão e nomes de classes
    conf_matrix = np.array([[50, 10], [5, 35]])
    class_names = ['Class 1', 'Class 2']

    # Plotar a matriz de confusão
    plot_confusion_matrix(conf_matrix, class_names)

    # Supondo que você tenha a importância das features
    feature_importance = np.array([0.2, 0.1, 0.4, 0.3])
    feature_names = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4']

    # Plotar a importância das features
    plot_feature_importance(feature_importance, feature_names)


if __name__ == '__main__':
    main()
