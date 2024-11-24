import os
from pathlib import Path
from typing import List, Union

from typing_extensions import deprecated


class FileHandler:
    def is_file_of_type(self, file: Union[Path, str], extension: str) -> bool:
        return Path(file).is_file() and str(file).lower().endswith(extension)

    @deprecated("Use is_file_of_type instead.")
    def is_mp4_file(self, file: Union[Path, str]):
        """Check if the file has a .mp4 extension."""
        return Path(file).is_file() and str(file).lower().endswith('.mp4')

    @deprecated("Use is_file_of_type instead.")
    def is_ts_file(self, file: Union[Path, str]):
        return Path(file).is_file() and str(file).lower().endswith('.ts')

    def get_file_modified_date(self, file_path: str) -> float:
        """Get the modification time of a file."""
        return os.path.getmtime(file_path)

    def get_number_of_files_in_dir(self, directory: Path, extension: str = "*") -> int:
        if not directory.is_dir():
            raise ValueError("The provided path is not a valid directory.")

        # Count the number of files in the folder
        if extension == "*":
            num_files = sum(1 for _ in directory.iterdir() if _.is_file())
        else:
            num_files = sum(1 for _ in directory.iterdir() if self.is_file_of_type(_, extension))

        return num_files

    def get_children_paths(self, directory: Path, extension: str = "*", recursive=False) -> List[Path]:
        """
        Gets all the files with the given extension in the directory.

        Args:
            directory: The directory to search in.
            extension: The extension of the files to search for.
            Defaults to '*', to search for all files.
            recursive: Whether to search subfolders recursively.

        Returns:
            A list of paths to the files with the given extension.

        Raises:
            ValueError: If the provided path is not a valid directory.
        """
        if not directory.is_dir():
            raise ValueError("The provided path is not a valid directory.")

        paths: List[Path] = []
        items = [directory]  # Initialize with the starting directory

        num_processed_files: int = 0

        while items:
            current_item = items.pop()
            if current_item.is_dir():
                if recursive or num_processed_files == 0:
                    # Add subdirectories to the list for iterative processing
                    items.extend(current_item.iterdir())
            elif extension == "*" or (current_item.suffix == extension):
                paths.append(current_item)

            num_processed_files += 1

        return paths

    def save_dict_to_file(self, data: dict, file_path: Path, header: str = None):
        with open(file_path, 'w') as file:
            if header is not None:
                file.write(f"{header}\n\n")

            for key, value in data.items():
                file.write(f"{key}: {value}\n")
