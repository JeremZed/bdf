import argparse
import os
from bdf.ui import DataVizApp


def main():
    parser = argparse.ArgumentParser(description="Gestion des commandes pour bdf.")
    subparsers = parser.add_subparsers(dest="command", help="Sous-commandes disponibles.")

    run_parser = subparsers.add_parser("run", help="Exécuter une application.")
    run_parser.add_argument("app", help="Nom de l'application à exécuter.")

    args = parser.parse_args()

    if args.command == "run":
        if args.app == "app":
            app = DataVizApp(name="DataViz")
            app.run(debug=True)
        else:
            print(f"Action '{args.app}' inconnue.")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()