import json
import queue
import socket
import threading
import time
from datetime import datetime
from msilib import gen_uuid
from pathlib import Path
from typing import List, Callable, TextIO

from .argument_model import ArgumentModel
from .message_model import MessageModel
from .message_payload import MessagePayload
from ..logging import HoornLogger


class Connector:
	"""
	Low-Level API for connecting with remote hosts.
	"""
	def __init__(self, logger: HoornLogger, message_received_listener: Callable[[MessageModel], None], module_separator: str = "Common.Connector", end_of_message_token: str = "<eom>"):
		self._logger = logger
		self._message_received_listener: Callable[[MessageModel], None] = message_received_listener

		self._module_separator = module_separator
		self._shutdown_signal: threading.Event = threading.Event()
		self._end_of_message_token = end_of_message_token
		self._socket: socket = None

	def shutdown(self):
		self._logger.debug("Stopping listening loop because of shutdown signal.", separator=self._module_separator)
		self._shutdown_signal.set()

		time.sleep(1)  # Wait for the processing thread to finish
		if self._socket is not None:
			script_path: Path = Path(__file__).parent.parent.joinpath('unregister.json')

			with open(script_path, 'r') as f:
				message: bytes = self._encode_message(f)
				self._logger.debug("Unregistering Component from Middleman", separator=self._module_separator)
				self._socket.sendall(message)
				time.sleep(1)  # Wait for the unregister message to send.

			self._socket.close()
			self._socket = None

	def connect_to_remote(self, host: str, port: int, component_port: int = 5555) -> socket:
		"""Connects to a remote host and port using TCP and continuously listens for data.

		Args:
		   host: The hostname or IP address of the remote host.
		   port: The port number on the remote host.
		   component_port: The port number to bind the socket to. Defaults to 5555.

		Returns:
		   None. Logs error messages on failure.
		"""
		s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		s.bind((host, component_port))
		try:
			s.connect((host, port))
		except ConnectionRefusedError:
			self._logger.error(f"Connection refused by {host}:{port}", separator=self._module_separator)
			return None
		self._logger.info(f"Connected to {host}:{port}", separator=self._module_separator)

		reading_thread = threading.Thread(target=self.read_data_loop, args=(s, host, port, self._shutdown_signal))
		reading_thread.start()

		keep_alive_thread = threading.Thread(target=self._keep_alive_loop, args=[self._shutdown_signal])
		keep_alive_thread.start()

		self._socket = s
		return s

	def read_data_loop(self, s: socket, host: str, port: int, shutdown_signal: threading.Event):
		buffer = b""
		message_queue = queue.Queue()  # Create a queue for messages

		# Start a separate thread for processing messages
		processing_thread = threading.Thread(target=self._process_messages, args=(message_queue, shutdown_signal))
		processing_thread.daemon = True
		processing_thread.start()

		while not shutdown_signal.is_set():
			try:
				data = s.recv(4096)
				if not data:
					self._logger.info(f"Connection closed by {host}:{port}", separator=self._module_separator)
					break

				buffer += data

				if self._end_of_message_token.encode() in buffer:
					message, remaining = buffer.split(self._end_of_message_token.encode(), 1)
					message = message.decode()
					message_queue.put((message, host, port))  # Put message in the queue
					buffer = remaining

			except socket.timeout:
				self._logger.warning(f"Timeout while receiving data from {host}:{port}", separator=self._module_separator)
			except OSError as e:
				if shutdown_signal.is_set():
					break

				self._logger.error(f"Error receiving data from {host}:{port}: {e}", separator=self._module_separator)
				break

		# Signal the processing thread to stop
		message_queue.put(None)
		processing_thread.join()

	def _keep_alive_loop(self, shutdown_signal: threading.Event) -> None:
		script_path: Path = Path(__file__).parent.parent.joinpath('keep_alive.json')

		with open(script_path, 'r') as f:
			message: bytes = self._encode_message(f)

		time.sleep(30) # No need to send a keep-alive message for the first 5 seconds

		while not shutdown_signal.is_set():
				self._socket.sendall(message)
				time.sleep(30)


	def _process_messages(self, message_queue: queue.Queue, shutdown_signal: threading.Event):
		while not shutdown_signal.is_set():
			try:
				item = message_queue.get(timeout=1)  # Get message from the queue with timeout
				if item is None:  # Check for termination signal
					break

				message, host, port = item
				self._process_message(message)
				message_queue.task_done()  # Indicate that the message has been processed
			except queue.Empty:
				pass  # Handle empty queue (timeout)

	def _process_message(self, data: str) -> None:
		# self._logger.debug(f"Received from {host}:{port}: {data}", separator=self._module_separator)
		json_data = json.loads(data)

		message_payload = json_data["payload"]
		args: List[ArgumentModel] = []

		for arg_data in message_payload["args"]:
			args.append(ArgumentModel(
				type=arg_data["type"],
				value=arg_data["value"]
			))

		processed_message_payload: MessagePayload = MessagePayload(
			action=message_payload["action"],
			args=args
		)

		processed_message: MessageModel = MessageModel(
			target_uuid=json_data["target_id"],
            payload=processed_message_payload
        )

		self._message_received_listener(processed_message)

	def _encode_message(self, file: TextIO) -> bytes:
		message = json.load(file)
		message["time_sent"] = str(datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:-3] + "Z" )
		message["unique_id"] = gen_uuid()
		message = json.dumps(message)
		message += self._end_of_message_token
		byte_buffer = bytes(message, encoding='utf-8')
		return byte_buffer
