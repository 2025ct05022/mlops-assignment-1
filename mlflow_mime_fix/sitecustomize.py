"""Register the JavaScript MIME type in MLflow's spawned server process."""

import mimetypes
import flask
from pathlib import Path

# Initialize from the Windows MIME registry first; otherwise a later lazy
# `mimetypes.init()` can overwrite this registration with `text/plain`.
mimetypes.init()
mimetypes.add_type("text/javascript", ".js")

_original_send_from_directory = flask.send_from_directory


def _send_from_directory_with_javascript_mime(directory, path, *args, **kwargs):
	response = _original_send_from_directory(directory, path, *args, **kwargs)
	if str(path).lower().endswith(".js"):
		response.mimetype = "text/javascript"
	return response


flask.send_from_directory = _send_from_directory_with_javascript_mime

_original_process_response = flask.Flask.process_response


def _process_response_with_javascript_mime(self, response):
	response = _original_process_response(self, response)
	if flask.request.path.lower().endswith(".js"):
		response.mimetype = "text/javascript"
	return response


flask.Flask.process_response = _process_response_with_javascript_mime
