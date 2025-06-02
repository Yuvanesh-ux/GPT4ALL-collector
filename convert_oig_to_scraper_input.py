import argparse
import jsonlines
import os
import sys

def validateFilePath(filePath, allowedBaseDir, mode):
    """
    Validates that the given filePath:
        - Is not absolute
        - Does not contain '..' as a path component
        - Is located within allowedBaseDir
    Returns the absolute, normalized path in allowedBaseDir if valid.
    Exits the script if invalid.
    """
    if os.path.isabs(filePath):
        print(f"Error: Absolute paths are not allowed: {filePath}")
        sys.exit(1)
    normPath = os.path.normpath(filePath)
    if normPath.startswith("..") or os.path.isabs(normPath) or any(part == ".." for part in normPath.split(os.sep)):
        print(f"Error: Parent directory traversal is not allowed in path: {filePath}")
        sys.exit(1)
    absAllowedDir = os.path.abspath(allowedBaseDir)
    absFilePath = os.path.abspath(os.path.join(absAllowedDir, normPath))
    # Ensure the final resolved path is within the allowed directory
    if not absFilePath.startswith(absAllowedDir + os.sep) and absFilePath != absAllowedDir:
        print(f"Error: Path escapes allowed directory: {filePath}")
        sys.exit(1)
    # For reading, ensure the file actually exists
    if mode == "r" and not os.path.exists(absFilePath):
        print(f"Error: File does not exist: {filePath}")
        sys.exit(1)
    # For writing, ensure the directory exists
    if mode == "w":
        outputDir = os.path.dirname(absFilePath)
        if not os.path.exists(outputDir):
            print(f"Error: Output directory does not exist: {outputDir}")
            sys.exit(1)
        if not os.path.isdir(outputDir):
            print(f"Error: Output path parent is not a directory: {outputDir}")
            sys.exit(1)
    return absFilePath

# Set the base directory for allowed file operations
ALLOWED_BASE_DIR = "data"

# create an ArgumentParser object and define the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("input_file", help="path to the input JSON file (restricted under ./data/)")
parser.add_argument("output_file", help="path to the output JSONLINES file (restricted under ./data/)")
args = parser.parse_args()

inputFilePath = validateFilePath(args.input_file, ALLOWED_BASE_DIR, "r")
outputFilePath = validateFilePath(args.output_file, ALLOWED_BASE_DIR, "w")

# open the input file and read the data
with open(inputFilePath, "r") as input_file:
    input_data = jsonlines.Reader(input_file)

    # open the output file and write the output data as a JSONLINES file
    with open(outputFilePath, "w") as output_file:
        output_data = jsonlines.Writer(output_file)

        # process each JSON object in the input data
        for json_obj in input_data:
            # extract the prompt and source fields
            prompt = json_obj["text"]
            source = json_obj["metadata"]["source"]

            # remove the "<human>:" and "<bot>:" tags and everything that comes after them
            prompt = prompt.split("<human>: ")[-1]
            prompt = prompt.split("<bot>:")[0]

            # strip leading and trailing whitespace from the prompt
            prompt = prompt.strip()

            # create a new object with the processed data and add it to the output file
            output_obj = {"00": prompt, "source": source}
            output_data.write(output_obj)