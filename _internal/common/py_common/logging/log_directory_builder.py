import os
from pathlib import Path
from typing import List


class LogDirectoryBuilder:
	"""
	Class to build a log directory structure based on given criteria.
	"""

	@staticmethod
	def build_log_directory(app_name: str, extra_dirs: List[str]) -> Path:
		"""
		Builds a log directory structure based on the given application name and extra directories.

		This function constructs a path for storing log files. It starts from the user's local directory,
		adds the application name, then appends any extra directories specified, and finally adds a "Logs" subdirectory.

		:param app_name: The name of the application, used as the first subdirectory name.
		:param extra_dirs: A list of additional directory names to be appended to the path.

		:returns Path: A Path object representing the constructed log directory path.
		"""
		local_dir = LogDirectoryBuilder._get_user_local_directory()
		app_dir = local_dir.joinpath(app_name)
		log_dir = app_dir

		for edir in extra_dirs:
			log_dir = log_dir.joinpath(edir)

		log_dir.joinpath("Logs")

		return log_dir


	@staticmethod
	def _get_user_local_directory() -> Path:
		"""
		This function retrieves the user's local directory path.

		The function attempts to expand the user's home directory using the `os.path.expanduser` function.
		If an exception occurs during this process, it is raised.

		After getting the user's home directory, the function constructs the local directory path by joining
		the home directory with "AppData" and "Local".

		:returns Path: The constructed path to the user's local directory.
		"""
		try:
			user_config_dir = os.path.expanduser("~")
		except Exception as e:
			raise e

		local_dir = os.path.join(user_config_dir, "AppData", "Local")
		return Path(local_dir)

