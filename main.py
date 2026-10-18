import argparse

import classification
import regression
import tests


def main():
    parser = argparse.ArgumentParser(description="perceptron wielowarstwowy, projekt 1")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("test", help="sprawdza forward i backward na modelu z wykladu")

    train = commands.add_parser("train", help="trenuje siec i zapisuje wyniki")
    train.add_argument("task", choices=["regression", "classification"])

    args = parser.parse_args()

    if args.command == "test":
        tests.test_forward_backward()
    elif args.command == "train":
        if args.task == "regression":
            regression.run()
        else:
            classification.run()


if __name__ == "__main__":
    main()
