import argparse
import jsonlines
from nomic import atlas

# --- Validation configuration ---
ALLOWED_FIELDS = {"00", "source"}
REQUIRED_FIELDS = {"00"}
INDEXED_FIELD = "00"
MAX_FIELD_LENGTH = 4096   # bytes per field value
MAX_RECORD_LENGTH = 8192  # bytes per record (all fields)
MAX_RECORDS = 10000

def isValidAtlasRecord(record, fileName):
    """
    Validate that a record:
    - Is a dict with only allowed fields.
    - Has all required fields, and they are non-empty strings of acceptable length.
    - All fields are strings and not longer than MAX_FIELD_LENGTH bytes.
    - Record size (all fields) not exceeding MAX_RECORD_LENGTH.
    """
    if not isinstance(record, dict):
        return False
    # Check allowed fields
    if not set(record.keys()).issubset(ALLOWED_FIELDS):
        return False
    # Check required fields
    for required in REQUIRED_FIELDS:
        if required not in record:
            return False
        value = record[required]
        if not isinstance(value, str) or not value.strip():
            return False
        if len(value.encode('utf-8')) > MAX_FIELD_LENGTH:
            return False
    # Per field validation
    totalBytes = 0
    for key, value in record.items():
        # Only string values allowed
        if not isinstance(value, str):
            return False
        fieldLen = len(value.encode('utf-8'))
        if fieldLen > MAX_FIELD_LENGTH:
            return False
        totalBytes += fieldLen
    if totalBytes > MAX_RECORD_LENGTH:
        return False
    return True

def sanitizeAtlasRecord(record, fileName):
    """
    Returns a sanitized, validated atlas record:
    - Drops disallowed fields.
    - Fills missing 'source' as needed.
    """
    sanitized = {}
    for field in ALLOWED_FIELDS:
        if field in record and isinstance(record[field], str):
            sanitized[field] = record[field]
    # Fill 'source'
    if 'source' not in sanitized:
        sanitized['source'] = fileName
    return sanitized

def loadAndValidateRecords(inputFileName):
    """
    Loads, validates, and sanitizes records up to MAX_RECORDS from a JSONL file.
    Invalid or over-limit records are skipped.
    """
    validData = []
    with jsonlines.open(inputFileName, "r") as inputFile:
        for obj in inputFile:
            if len(validData) >= MAX_RECORDS:
                break
            if not isinstance(obj, dict):
                continue
            # Drop extraneous fields and add 'source' if missing
            sanitized = sanitizeAtlasRecord(obj, inputFileName)
            if isValidAtlasRecord(sanitized, inputFileName):
                validData.append(sanitized)
    return validData

# create an ArgumentParser object and define the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("input_file", help="path to the input json file")
args = parser.parse_args()

# load and validate data
data = loadAndValidateRecords(args.input_file)

print(f"processing {len(data)} items")

# index the prompt field using atlas.map_text()
project = atlas.map_text(data=data,
                         indexed_field=INDEXED_FIELD,
                         name=args.input_file,
                         colorable_fields=["source"]
                        )