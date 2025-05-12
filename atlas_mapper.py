import argparse
import jsonlines
from nomic import atlas
import sys

# Validation parameters
MAX_RECORDS = 10000
MAX_FIELD_LENGTH = 8192
ALLOWED_FIELDS = {'00', 'source'}
REQUIRED_FIELD = '00'
MAX_FILE_SIZE_MB = 50  # safeguard against giant files

def validate_record(obj, idx):
    # Record must be a dict
    if not isinstance(obj, dict):
        print(f"Warning: Skipping record {idx}: not a JSON object.", file=sys.stderr)
        return None

    # Required field check
    if REQUIRED_FIELD not in obj:
        print(f"Warning: Skipping record {idx}: missing required field '{REQUIRED_FIELD}'.", file=sys.stderr)
        return None

    val = obj[REQUIRED_FIELD]
    if not isinstance(val, str):
        print(f"Warning: Skipping record {idx}: field '{REQUIRED_FIELD}' must be a string.", file=sys.stderr)
        return None

    # Basic text checks (non-empty, not all whitespace, not too long)
    if not val.strip():
        print(f"Warning: Skipping record {idx}: '{REQUIRED_FIELD}' field is empty or whitespace.", file=sys.stderr)
        return None

    if len(val) > MAX_FIELD_LENGTH:
        print(f"Warning: Skipping record {idx}: '{REQUIRED_FIELD}' field too long.", file=sys.stderr)
        return None

    # No control chars (except \n, \t) or newlines inside text
    if any(ord(c) < 32 and c not in '\n\t' for c in val):
        print(f"Warning: Skipping record {idx}: '{REQUIRED_FIELD}' contains control characters.", file=sys.stderr)
        return None

    # Only allow expected fields (for now), and only strings as values (except 'source')
    filtered = {}
    for k in obj:
        if k not in ALLOWED_FIELDS:
            print(f"Warning: Skipping record {idx}: unexpected field '{k}'.", file=sys.stderr)
            return None
        v = obj[k]
        if k == 'source':
            if not isinstance(v, str):
                print(f"Warning: Skipping record {idx}: 'source' field must be a string.", file=sys.stderr)
                return None
            if len(v) > 256:
                print(f"Warning: Truncating 'source' field in record {idx} to 256 characters.", file=sys.stderr)
                v = v[:256]
            filtered[k] = v
        elif k == REQUIRED_FIELD:
            filtered[k] = v
    return filtered

# create an ArgumentParser object and define the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("input_file", help="path to the input json file")
args = parser.parse_args()

# Pre-check: Warn and abort on giant files (DoS safeguard)
import os
try:
    file_size_mb = os.path.getsize(args.input_file) / (1024*1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        print(f"Error: input file size {file_size_mb:.1f} MB exceeds maximum {MAX_FILE_SIZE_MB} MB. Aborting.", file=sys.stderr)
        sys.exit(1)
except OSError as e:
    print(f"Error: Cannot stat input file: {e}", file=sys.stderr)
    sys.exit(1)

# load the data from the input file using jsonlines, with validation
data = []
with jsonlines.open(args.input_file, "r") as input_file:
    for idx, obj in enumerate(input_file):
        if len(data) >= MAX_RECORDS:
            print(f"Warning: Maximum of {MAX_RECORDS} records loaded; additional records skipped.", file=sys.stderr)
            break
        # Attach source if not present (truncate if needed)
        if 'source' not in obj:
            src_val = args.input_file
            if len(src_val) > 256:
                src_val = src_val[:256]
            obj['source'] = src_val
        obj_valid = validate_record(obj, idx)
        if obj_valid is not None:
            data.append(obj_valid)

if not data:
    print("Error: No valid records to process after validation. Exiting.", file=sys.stderr)
    sys.exit(1)

print(f"processing {len(data)} items")

# index the prompt field using atlas.map_text()
project = atlas.map_text(data=data,
                         indexed_field="00",
                         name=args.input_file,
                         colorable_fields=["source"]
                        )