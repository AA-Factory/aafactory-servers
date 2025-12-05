import logging
import os
import subprocess

import base64
import shutil
import magic
import mimetypes


def b64_to_bytes(data: str) -> bytes:
    return base64.b64decode(data)


def bytes_to_b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")

def detect_file_extension(file_bytes: bytes) -> str:
    """
    Detect the file extension from bytes using magic numbers.
    Falls back to mimetypes guess or default_ext.
    """
    mime = magic.from_buffer(file_bytes, mime=True)  # e.g., 'image/png', 'video/mp4'
    ext = mimetypes.guess_extension(mime)
    if ext:
        return ext.lstrip(".")  # remove leading dot
    raise ValueError(f"Could not determine file extension for MIME type: {mime}")


def write_bytes_to_path(data: bytes, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def run_python_command(command_args: list, cwd: str, env: dict = None) -> str:
    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    logging.getLogger().info(f"Executing command: {' '.join(command_args)} in CWD: {cwd}")
    try:
        res = subprocess.run(
            command_args,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            env=full_env,
        )
        if res.stdout:
            logging.getLogger().debug(res.stdout)
        if res.stderr:
            logging.getLogger().warning(res.stderr)
        return res.stdout
    except subprocess.CalledProcessError as e:
        logging.getLogger().error(
            f"Command failed: returncode={e.returncode}. stdout={e.stdout} stderr={e.stderr}"
        )
        raise RuntimeError(f"Python command failed: {e.stderr or e.stdout}")
    except FileNotFoundError:
        logging.getLogger().error("Executable or script not found.")
        raise RuntimeError("Command executable or script not found.")
    except Exception as e:
        logging.getLogger().exception("Unexpected error running command")
        raise RuntimeError(f"Unexpected error: {e}")


def delete_files_from_folder(folder_path: str) -> None:
    # Remove every entry inside the folder but keep the folder itself.
    for name in os.listdir(folder_path):
        path = os.path.join(folder_path, name)
        try:
            # If it's a file or symlink, remove it; if it's a directory, rmtree it.
            if os.path.islink(path) or os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
            else:
                # Unknown file type: try removing as file.
                os.remove(path)
            logging.getLogger().info(f"Removed: {path}")
        except Exception:
            logging.getLogger().warning(f"Could not remove: {path}", exc_info=True)