import argparse

def parse_arguments():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Samvara-AI Debugger")

    # Add debug mode argument
    parser.add_argument('-x', '--debug', action='store_true', help="Run in debug mode")

    # Add screen name argument
    parser.add_argument('-s', '--screen', type=str, help="Specify screen name")

    # Add help argument (note: argparse automatically handles --help)

    # Parse arguments and return them
    return parser.parse_args()
