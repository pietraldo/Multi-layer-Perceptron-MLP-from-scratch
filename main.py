import argparse

import tests
from configs import EXPERIMENTS
from experiments.runner import run_experiment


def main():
    parser = argparse.ArgumentParser(description="perceptron wielowarstwowy, projekt 1")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("test", help="sprawdza forward i backward na modelu z wykladu")
    commands.add_parser("list", help="wypisuje eksperymenty z configs.py")

    run = commands.add_parser("run", help="uruchamia eksperymenty z configs.py i zapisuje wyniki")
    run.add_argument("names", nargs="+", choices=list(EXPERIMENTS), metavar="name")
    run.add_argument("--seeds", type=int, nargs="+", help="nadpisuje ziarna z konfiguracji")
    run.add_argument("--epochs", type=int, help="nadpisuje liczbe epok, np. do szybkiego sprawdzenia")

    args = parser.parse_args()

    if args.command == "test":
        tests.test_forward_backward()
    elif args.command == "list":
        for name, e in EXPERIMENTS.items():
            print(f"{name:30} {len(e.variants)} variants x {len(e.seeds)} seeds  {e.description}")
    else:
        for name in args.names:
            run_experiment(EXPERIMENTS[name], seeds=args.seeds, epochs=args.epochs)


if __name__ == "__main__":
    main()
