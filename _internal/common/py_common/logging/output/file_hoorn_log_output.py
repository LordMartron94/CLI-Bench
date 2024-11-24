import os
from pathlib import Path
from typing import List

from ..reserved_keys import RESERVED_LOGGING_KEYS
from ...handlers.file_handler import FileHandler
from ...logging.formatting.log_text_formatter import HoornLogTextFormatter
from ...logging.hoorn_log import HoornLog
from ...logging.output.hoorn_log_output_interface import HoornLogOutputInterface


class FileHoornLogOutput(HoornLogOutputInterface):
    def __init__(self, log_directory: Path, max_logs_to_keep: int = 3, create_directory: bool = True, use_combined: bool = True, max_separator_length: int = 30):
        """
        Formats logs into text files and keeps track of the maximum number of logs to keep.

        :param log_directory: The base directory for logs.
        :param max_logs_to_keep: The max number of logs to keep (per directory).
        :param create_directory: Whether to initialize the creation of log directories if they don't exist.
        :param use_combined: Whether to combine multiple separators also into a single log file.
        :param max_separator_length: The maximum length of the separator in the log file names.
        """

        self._file_handler: FileHandler = FileHandler()

        self._root_log_directory: Path = log_directory
        self._max_logs_to_keep: int = max_logs_to_keep
        self._use_combined: bool = use_combined
        self._max_separator_length: int = max_separator_length

        self._validate_directory(self._root_log_directory, create_directory)

        self._increment_logs()

        super().__init__(is_child=True)

    def _validate_directory(self, directory: Path, create_directory: bool):
        if not directory.exists():
            if create_directory:
                directory.mkdir(parents=True, exist_ok=True)
                return

            raise FileNotFoundError(f"Log directory {directory} does not exist")

    def _increment_logs(self) -> None:
        """
        Increments the log number by 1 and removes old logs if necessary.

        :return: None
        """

        children = self._file_handler.get_children_paths(self._root_log_directory, ".txt", recursive=True)
        children.sort(reverse=True)

        organized_by_separator: List[List[Path]] = self._organize_logs_by_subdirectory(children)

        for directory_logs in organized_by_separator:
            self._increment_logs_in_directory(directory_logs)

    def _increment_logs_in_directory(self, log_files: List[Path]) -> None:
        for i in range(len(log_files)):
            child = log_files[i]
            number = int(child.stem.split("_")[-1])
            if number + 1 > self._max_logs_to_keep:
                os.remove(child)
                continue

            os.rename(child, Path.joinpath(child.parent.absolute(), f"log_{number + 1}.txt"))

    def _get_path_to_log_to(self, separator: str = None):
        directory = self._root_log_directory

        if separator is not None and separator != "":
            directory = self._root_log_directory.joinpath(separator)
            self._validate_directory(directory, True)

        return Path.joinpath(directory, f"log_1.txt")

    def _get_number_of_logs_in_directory(self, log_directory: Path) -> int:
        return self._file_handler.get_number_of_files_in_dir(log_directory, ".txt")

    def _write_log(self, formatted_log: str, separator: str = None, encoding: str = 'utf-8') -> None:
        log_file = self._get_path_to_log_to(separator)

        for reserved_key in RESERVED_LOGGING_KEYS:
            formatted_log = formatted_log.replace(reserved_key, "")

        with open(log_file, "a", encoding=encoding) as f:
            f.write(formatted_log + "\n")

    def output(self, hoorn_log: HoornLog, encoding="utf-8") -> None:
        formatter: HoornLogTextFormatter = HoornLogTextFormatter()
        formatted_log: str = formatter.format(hoorn_log)

        self._write_log(formatted_log, separator=hoorn_log.separator, encoding=encoding)

        if self._use_combined and hoorn_log.separator is not None and hoorn_log.separator != "":
            self._handle_combined(hoorn_log, encoding)

    def _handle_combined(self, hoorn_log: HoornLog, encoding) -> None:
        formatter: HoornLogTextFormatter = HoornLogTextFormatter()
        formatted_log: str = f"[{hoorn_log.separator:<{self._max_separator_length}}] " + formatter.format(hoorn_log)
        self._write_log(formatted_log, separator=None, encoding=encoding)

    def _organize_logs_by_subdirectory(self, log_paths: List[Path]) -> List[List[Path]]:
        """
        Organizes a list of log file paths into a list of lists,
        where each sublist contains logs from the same subdirectory.

        Args:
          log_paths: A list of WindowsPath objects representing log file paths.

        Returns:
          A list of lists, where each sublist contains log paths from the same subdirectory.
        """
        log_groups = {}
        for log_path in log_paths:
            parent_dir = log_path.parent.name  # Get the name of the parent directory
            if parent_dir not in log_groups:
                log_groups[parent_dir] = []
            log_groups[parent_dir].append(log_path)
        return list(log_groups.values())
