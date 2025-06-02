import argparse
import jsonlines
from nomic import atlas

# --------------------------
# Validation Constants
# --------------------------
MAX_TEXT_LENGTH = 1024
MAX_SOURCE_LENGTH = 255
MAX_RECORDS = 10000
MAX_RECORD_SIZE = 4096  # bytes per record

def isPrintableString(s):
    """Check if s is a string with only printable chars."""
    if not isinstance(s, str):
        return False
    try:
        s.encode('utf-8')
    except Exception:
        return False
    return all(32 <= ord(c) <= 126 or c in '\t\n\r' for c in s)

def validateRecord(rec, inputFile):
    """
    Validates a JSONL record based on strict schema, type, and size checks.
    Returns (is_valid, error_msg) tuple.
    """
    if not isinstance(rec, dict):
        return False, "Record is not a dict."
    # Must have field '00' as string with length within bounds
    if "00" not in rec:
        return False, "'00' field missing."
    text = rec["00"]
    if not isinstance(text, str) or not text.strip():
        return False, "'00' field must be non-empty string."
    if len(text) > MAX_TEXT_LENGTH:
        return False, "'00' field exceeds max length."
    if not isPrintableString(text):
        return False, "'00' contains non-printable chars."
    # Source field optional, enforce length/type if present or add if missing
    if "source" not in rec:
        rec["source"] = inputFile
    if not isinstance(rec["source"], str):
        rec["source"] = str(rec["source"])
    if len(rec["source"]) > MAX_SOURCE_LENGTH:
        rec["source"] = rec["source"][:MAX_SOURCE_LENGTH]
    # Check record size
    try:
        rawSize = len(str(rec).encode('utf-8'))
        if rawSize > MAX_RECORD_SIZE:
            return False, "Record size exceeds limit."
    except Exception:
        return False, "Error encoding record."
    return True, ""

def loadValidatedData(inputFile):
    """
    Loads and validates data from JSONL file.
    Returns a list of valid records.
    Provides record and skipped count logging.
    """
    data = []
    skipped = 0
    with jsonlines.open(inputFile, "r") as fileHandle:
        for idx, obj in enumerate(fileHandle):
            if idx >= MAX_RECORDS:
                print(f"Maximum record limit {MAX_RECORDS} reached; stopping.")
                break
            isValid, err = validateRecord(obj, inputFile)
            if isValid:
                data.append(obj)
            else:
                skipped += 1
    if skipped > 0:
        print(f"Skipped {skipped} malformed records.")
    return data

# create an ArgumentParser object and define the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("input_file", help="path to the input json file")
args = parser.parse_args()

# load and validate the data from the input file
data = loadValidatedData(args.input_file)

print(f"processing {len(data)} items")

# index the prompt field using atlas.map_text()
project = atlas.map_text(data=data,
                         indexed_field="00",
                         name=args.input_file,
                         colorable_fields=["source"]
                        )