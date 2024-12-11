#!/usr/bin/env python3
from src.main import main

def main():
    """
    Main function to run the script.
    """
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
