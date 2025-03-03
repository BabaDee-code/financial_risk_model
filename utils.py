"""
utils.py

This module contains utility functions for logging, plotting, and other
common tasks that support the main modules in the project.
"""

import logging

def setup_logging(log_file: str = "app.log") -> None:
    """
    Sets up the logging configuration for the project.

    Args:
        log_file (str): The file path for the log file.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging is set up.")

if __name__ == "__main__":
    setup_logging()
    logging.info("Utility module logging test complete.")
