import argparse
import jsonlines
import os
import sys
import re

# List of banned words/phrases that typically indicate attempts at prompt injection, data poisoning, or unwanted jailbreaking
BANNED_PATTERNS = [
    r"\bjailbreak\b",
    r"\bpoison(ed|ing)?\b",
    r"\bmalicious\b",
    r"\bignore\s+all\s+previous\b",
    r"\bdo\s+anything\s+now\b",
    r"\bexploit\b",
    r"<\s*(system|user)\s*>:",
    r"[\x00-\x08\x0B-\x0C\x0E-\x1F]",  # Control chars except tab/newline/carriage-return
]

MAX_PROMPT_LEN = 4096
MIN_PROMPT_LEN = 1
MAX_SOURCE_LEN = 255
MIN_SOURCE_LEN = 1

def is_string_and_length(s, min_len, max_len):
    return isinstance(s, str) and min_len <= len(s.strip()) <= max_len

def contains_banned_patterns(text):
    for pat in BANNED_PATTERNS:
        if re.search(pat, text, flags=re.IGNORECASE):
            return True
    return False

def validate_json_obj(json_obj):
    # Validate "text"
    text = json_obj.get("text")
    if not is_string_and_length(text, MIN_PROMPT_LEN, MAX_PROMPT_LEN):
        return False, "Missing or invalid 'text' field"
    # Validate "metadata"
    metadata = json_obj.get("metadata")
    if not isinstance(metadata, dict):
        return False, "Missing or invalid 'metadata' (should be a dict)"
    source = metadata.get("source")
    if not is_string_and_length(source, MIN_SOURCE_LEN, MAX_SOURCE_LEN):
        return False, "Missing or invalid 'metadata.source' field"
    return True, ""

def is_safe_path(basedir, path, follow_symlinks=True):
    # Check if the resolved absolute path is under basedir and not a symlink
    if follow_symlinks:
        real_path = os.path.realpath(path)
    else:
        real_path = os.path.abspath(path)
    return os.path.commonpath([basedir, real_path]) == basedir

def is_symlink(path):
    try:
        return os.path.islink(path)
    except Exception:
        return False

def has_parent_ref(path):
    return '..' in os.path.normpath(path).split(os.path.sep)

def is_dotfile(path):
    base = os.path.basename(path)
    return base.startswith('.') and base not in ('.', '..')

def validate_path(file_path, mode='r'):
    cwd = os.path.abspath(os.getcwd())
    abs_path = os.path.abspath(file_path)

    # Disallow absolute paths
    if os.path.isabs(file_path):
        print(f"Error: Absolute paths are not allowed ('{file_path}')")
        sys.exit(1)

    # Disallow parent directory traversal
    if has_parent_ref(file_path):
        print(f"Error: Parent directory references ('..') are not allowed in path ('{file_path}')")
        sys.exit(1)

    # Ensure the file is within cwd
    if not is_safe_path(cwd, abs_path):
        print(f"Error: File path '{file_path}' escapes the working directory")
        sys.exit(1)

    # Disallow symlinks for both input/output
    if os.path.exists(abs_path) and is_symlink(abs_path):
        print(f"Error: File path '{file_path}' refers to a symbolic link")
        sys.exit(1)

    if mode == "w":
        # Disallow writing to hidden dotfiles for output
        if is_dotfile(file_path):
            print(f"Error: Output file '{file_path}' is a dotfile; writing to dotfiles is not allowed.")
            sys.exit(1)

    return abs_path

# create an ArgumentParser object and define the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("input_file", help="path to the input JSON file")
parser.add_argument("output_file", help="path to the output JSONLINES file")
args = parser.parse_args()

sanitized_input_file = validate_path(args.input_file, mode="r")
sanitized_output_file = validate_path(args.output_file, mode="w")

# open the input file and read the data
with open(sanitized_input_file, "r") as input_file:
    input_data = jsonlines.Reader(input_file)

    # open the output file and write the output data as a JSONLINES file
    with open(sanitized_output_file, "w") as output_file:
        output_data = jsonlines.Writer(output_file)

        # process each JSON object in the input data
        for json_obj in input_data:
            valid, msg = validate_json_obj(json_obj)
            if not valid:
                print(f"[WARN] Skipping record due to schema error: {msg}", file=sys.stderr)
                continue

            # extract the prompt and source fields
            prompt = json_obj["text"]
            source = json_obj["metadata"]["source"]

            # remove the "<human>:" and "<bot>:" tags and everything that comes after them
            prompt = prompt.split("<human>: ")[-1]
            prompt = prompt.split("<bot>:")[0]

            # strip leading and trailing whitespace from the prompt
            prompt = prompt.strip()

            # Re-validate after processing and apply content filtering
            if not is_string_and_length(prompt, MIN_PROMPT_LEN, MAX_PROMPT_LEN):
                print(f"[WARN] Skipping record due to empty or oversized prompt after preprocessing.", file=sys.stderr)
                continue
            if contains_banned_patterns(prompt) or contains_banned_patterns(source):
                print(f"[WARN] Skipping record due to banned pattern in prompt or source.", file=sys.stderr)
                continue

            # create a new object with the processed data and add it to the output file
            output_obj = {"00": prompt, "source": source}
            output_data.write(output_obj)