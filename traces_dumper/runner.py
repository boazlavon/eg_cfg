import os
import pty
import subprocess
import time
import select
import sys
import re
import ast
import linecache
from collections import defaultdict


END_MESSAGE1 = "trepan-xpy: That's all, folks..."
END_MESSAGE2 = (
    "The program finished - press enter to restart; anything else terminates."
)
PYTHON_LINE_TO_PC_OFFSETS = {}
PC_OFFSETS_TO_PYTHON_LINE_CACHE = {}
DELIMITER = "\n" + "#" * 30 + "\n"


# Function to send a command to trepan3k and get the output
def remove_ansi_escape_sequences(text):
    ansi_escape = re.compile(r"(?:\x1B[@-_][0-?]*[ -/]*[@-~])")
    text = ansi_escape.sub("", text)
    return text


def send_command(command, master_fd):
    os.write(master_fd, (command + "\n").encode())
    # time.sleep(0.1)  # Allow some time for output to be generated
    return read_output(master_fd)


# Function to read output using select for non-blocking I/O
def read_output(master_fd):
    output = []
    while True:
        # Use select to wait for data on the master side of the pty
        rlist, _, _ = select.select([master_fd], [], [], 0.1)
        if not rlist:
            break

        for fd in rlist:
            try:
                data = os.read(fd, 1024).decode()
                if data:
                    # output.append(data)
                    output.append(remove_ansi_escape_sequences(data))
            except OSError:
                return "".join(output)

    return "".join(output)


def run_trepan_auto(script_path):
    # Create a pseudoterminal to communicate with the subprocess
    master_fd, slave_fd = pty.openpty()

    # Start the trepan3k process using the pseudoterminal for stdin/stdout
    subprocess.Popen(
        ["trepan-xpy", script_path],
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        universal_newlines=True,
        bufsize=1,
    )
    step_output = ""
    while not step_output:
        step_output = read_output(master_fd)
    print(step_output)

    while True:
        # Command to step through the code
        step_output = send_command("step", master_fd)
        print(step_output)

        if END_MESSAGE2 in step_output:
            print("\nProgram finished, sending 'y' to restart.")
            send_command("y", master_fd)
            break

        if END_MESSAGE1 in step_output:
            break

    # Clean up file descriptors
    os.close(master_fd)
    os.close(slave_fd)


if __name__ == "__main__":
    # The path to the Python script you want to debug with trepan3k
    script_path = sys.argv[1]
    if not os.path.exists(script_path):
        raise IOError(f"Invalid script {script_path}")

    run_trepan_auto(script_path)
