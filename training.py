import torch
import torch.nn as nn
import numpy as np
from cnn import CNN
from data_loader import get_data


def evaluate_model(loader, model, criterion):
    correct_predictions = 0
    losses = []
    with torch.no_grad():
        for X_test, y_test in loader:
            y_pred = model(X_test)
            loss = criterion(y_pred, y_test)

            losses.append(loss)
            y_pred = torch.max(y_pred, 1)[1]
            correct_predictions += (y_pred == y_test).sum()

    accuracy = correct_predictions / len(loader) * 100
    return accuracy, np.mean(losses)


def train_model():
    model = CNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_loader, test_loader = get_data()

    epochs = 7

    for epoch in range(epochs):
        batch = 0
        train_correct_predictions = 0
        for X_train, y_train in train_loader:
            batch += 1
            y_pred = model(X_train)
            loss = criterion(y_pred, y_train)

            y_pred = torch.max(y_pred, 1)[1]
            train_correct_predictions += (y_pred == y_train).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 10 == 1:
                print(f'Learning batch {batch} of {len(train_loader)}')

        test_accuracy, test_loss = evaluate_model(test_loader, model, criterion)

        print(f'''
            \t Epoch {epoch + 1} of {epochs} completed: \n \t\t 
            train correct {train_correct_predictions}, 
            \n \t\t test correct {test_accuracy}, 
            \n \t\t test loss {test_loss}
        ''')


if __name__ == '__main__':
    train_model()
