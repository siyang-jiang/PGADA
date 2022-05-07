"""
Train a CNN using empirical risk minimization (non-episodic training).
"""

from src.erm_training_steps import get_data, get_model, train
from src.running_steps import prepare_output, set_and_print_random_seed


def main():
    prepare_output()
    set_and_print_random_seed()

    train_loader, val_loader, n_classes = get_data(two_stream=True)

    model = get_model(n_classes)

    train(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
