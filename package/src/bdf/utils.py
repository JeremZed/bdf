import os
from datetime import datetime
import random

import numpy as np

log_filename = "bdf.log"

class Tools:
    """
    Utility class for performing various operations.

    This class handles the display and saving of logs to files.
    It provides a method for logging messages with different verbosity levels,
    and the ability to set thresholds for displaying or writing logs to a file.

    Attributes:
        log_filename (str): Default name of the log file (bdf.log).
    """

    @staticmethod
    def log(message, level, show=True, write=True, threshold=1, pathfile=f"{os.getcwd()}{os.sep}{log_filename}", raising=True):
        """
            Displays and/or writes a log message based on its level and a defined threshold.

            This method allows logging a message based on its verbosity level. It can
            display the log in the console, write it to a file, or both, depending on the parameters.

            Args:
                message (str): The message to be logged.
                level (int): The verbosity level of the log (e.g., 1 for info, 2 for warning, etc.).
                show (bool, optional): If True, displays the message in the console. Defaults to True.
                write (bool, optional): If True, writes the message to a log file. Defaults to True.
                threshold (int, optional): The minimum verbosity level for the log to be displayed/written. Defaults to 1.
                pathfile (str, optional): The path of the file where to write the log. Defaults to "bdf.log" in the current directory.
                raising (bool, optional): If True, raises an exception when the path is incorrect or inaccessible. Defaults to True.

            Raises:
                FileNotFoundError: If the directory/file does not exist.

            Example:
                >>> Tools.log("A log message", level=1, show=True, write=True, threshold=1)
                "Displays the message in the console and writes it to the bdf.log file."
            """
        # If the log's verbosity level is greater than or equal to the threshold condition
        if level >= threshold:
            dt = datetime.now()
            time_of_log = dt.strftime("%y-%m-%d %H:%M:%S:%f")[:-3]
            prefix = f"[LOG] {time_of_log}|{level}| "
            content_to_log = f"{prefix}{message}\n"

            # Display the formatted log
            if show :
                print(content_to_log)

            # Write the message to a log file
            if write :
                # If only a directory path is provided
                # then create the default bdf.log file in that directory
                if os.path.isdir( pathfile )  :
                    os.path.join(pathfile, log_filename)

                if pathfile[-4:] == ".log":
                    dirname = os.path.dirname(pathfile)
                    # Check if the directory passed as a parameter exists
                    if os.path.exists(dirname) :
                        with open(pathfile, "+a") as f:
                            f.write(content_to_log)
                    else:
                        if raising :
                            raise FileNotFoundError(f"Directory not found: {dirname}")
                else:
                    if raising :
                        raise FileNotFoundError(f"Path not found: {pathfile}")

    def random_id(length=12, pattern=None, uids=None, chars_special=None, excludes_chars=None):
        """
            Generates a random identifier according to specified criteria.

            Args:
                length (int, optional):
                    Length of the generated identifier. Defaults to 12.
                pattern (list, optional):
                    Template to use for generating the identifier. Each character in the template can correspond to:
                        - "%s": lowercase letter.
                        - "%S": uppercase letter.
                        - "%d": digit.
                        - "%x": special character.
                        Any other character is used as is.
                    If `None`, a standard identifier is generated without constraints.
                uids (list, optional):
                    List of existing identifiers to avoid duplicates. Defaults to an empty list.
                chars_special (string, optional):
                    Set of usable special characters. Defaults to `"-_!?@$*."`.
                excludes_chars (list, optional):
                    List of characters to exclude during generation. Defaults to an empty list.

            Returns:
                str: Generated identifier that meets the specified criteria.

            Raises:
                RecursionError: If a unique identifier cannot be generated after several attempts.
                ValueError: If a parameter is not of the correct type.

            Notes:
                - If no `pattern` is provided, the identifier is generated using a mix of uppercase,
                lowercase, digits, and special characters.
                - Characters excluded in `excludes_chars` are never included in the generated identifier.
                - If the generated identifier already exists in `uids`, the function is called recursively
                to try to generate a new unique identifier.
        """

        sequence_chars_upper = "ABCDEFGHIJKLMNOPQRSTUVWXTZ"
        sequence_chars_lower = sequence_chars_upper.lower()
        sequence_digits = "0123456789"
        sequence_chars_special = chars_special if chars_special is not None else "-_!?@$*."

        if not isinstance(sequence_chars_special, str):
            raise ValueError("the chars_special parameter must be a string.")

        if chars_special is not None:
            if not all(char.isprintable() and not char.isspace() for char in chars_special):
                raise ValueError("the chars_special parameter contains invalid characters (spaces or non-printable).")

        if uids is None:
            uids = []

        if excludes_chars is None:
            excludes_chars = []

        if not isinstance(length, int) or length <= 0:
            raise ValueError("the length parameter must be a positive integer.")

        if not isinstance(uids, list):
            raise ValueError("the uids parameter must be a list.")

        if not isinstance(excludes_chars, list):
            raise ValueError("the excludes_chars parameter must be a list.")


        sequence_chars = sequence_chars_upper + sequence_chars_lower + sequence_digits + sequence_chars_special
        id = ""

        diff_chars = set(sequence_chars) - set(excludes_chars)
        if not diff_chars:
            raise ValueError("All possible characters are excluded. Cannot generate an identifier.")

        # Generation of a standard random id respecting the size passed as a parameter
        if pattern is None:
            for i in range(length):
                ch = Tools.get_random_element(sequence_chars, excludes_chars)
                id = id + ch
        else:

            if not isinstance(pattern, list) or len(pattern) == 0:
                raise ValueError("The pattern parameter must be a list containing at least one element.")

            all_sequences = {
                "%s": sequence_chars_lower,
                "%S": sequence_chars_upper,
                "%d": sequence_digits,
                "%x": sequence_chars_special
            }

            for c in pattern:
                if c in all_sequences:
                        ch = Tools.get_random_element(all_sequences[c], excludes_chars)
                else:
                    ch = c

                id = id + ch

        uids_set = set(uids)
        # We check if the generated id is not an id already present in the list
        if id in uids_set:
            return Tools.random_id(length, pattern, uids, chars_special, excludes_chars)

        return id

    def get_random_element(items, excludes=[]):
        """
            Returns a random element from the items list, excluding unauthorized elements.
        """
        elements = [item for item in items if item not in excludes]

        if not elements:
            raise ValueError("No element is available for selection.")

        return random.choice(elements)

    def is_all_numeric(df):
        """
            Checks if all columns of a DataFrame are of a numerical type.

            This function inspects the data types of the columns of a DataFrame to determine
            if they are all of a numerical type (e.g., int, float, etc.).

            Args:
                df (pd.DataFrame): The DataFrame to be analyzed.

            Returns:
                bool:
                    - `True` if all columns of the DataFrame are of a numerical type.
                    - `False` otherwise.

            Example:
                >>> import pandas as pd
                >>> import numpy as np
                >>> df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4.5, 5.6, 6.7]})
                >>> is_all_numeric(df1)
                True
                >>> df2 = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
                >>> is_all_numeric(df2)
                False
            """

        numeric_columns = df.select_dtypes(include=np.number).columns
        return len(numeric_columns) == len(df.columns)

    def all_features_present(features, columns):
        """
        Checks if all specified features are present in the list of columns.

        Args:
            features (list[str]): List of features to check.
            columns (list[str]): List of available columns.

        Returns:
            bool: True if all features are present, False otherwise.
        """
        return all(f in columns for f in features)
