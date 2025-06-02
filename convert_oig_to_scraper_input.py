import argparse
import jsonlines
import sys
import string

MAX_PROMPT_LENGTH = 2048  # could be configured in a config file if desired
TOXIC_KEYWORDS = {
    # Minimal set; in production, use comprehensive lexicon
    "poison", "kill", "hate", "attack", "xxxx", "f***", "<script>", "</script>",
    "drop table", "backdoor", "prompt injection", "sudo rm", "password", "exploit"
}
INJECTION_MARKERS = {"<bot>:", "<human>:", "<system>:", "<|", "|>"}

def isPrintable(s):
    """
    Returns True if all characters in s are printable (except common whitespace).
    """
    for c in s:
        if c not in string.printable and c not in '\n\r\t':
            return False
    return True

def containsToxicKeyword(text):
    """
    Returns True if any token from TOXIC_KEYWORDS is present (case-insensitive).
    """
    textLower = text.lower()
    for keyword in TOXIC_KEYWORDS:
        if keyword in textLower:
            return True
    return False

def containsPossibleInjection(text):
    """
    Returns True if the text contains likely prompt-injection tags/markers.
    """
    textLower = text.lower()
    for marker in INJECTION_MARKERS:
        if marker in textLower:
            return True
    return False

def sanitizePrompt(prompt):
    """
    Validates, filters, and sanitizes the user's prompt.
    Returns cleaned prompt, or None if validation fails.
    """
    # Remove "<human>: " and everything before, then "<bot>:" and everything after
    prompt = prompt.split("<human>: ")[-1]
    prompt = prompt.split("<bot>:")[0]
    prompt = prompt.strip()

    # Filter empty or too long prompts
    if not prompt or len(prompt) == 0:
        return None
    if len(prompt) > MAX_PROMPT_LENGTH:
        return None

    # Remove non-printable/control characters except whitespace
    promptSafe = "".join(c for c in prompt if (c in string.printable or c == '\n' or c == '\t'))
    promptSafe = promptSafe.strip()

    if not promptSafe or len(promptSafe) == 0:
        return None

    # Check for basic toxic/abusive/backdoor keywords
    if containsToxicKeyword(promptSafe):
        return None

    # Check for likely prompt injection markers, embedded tags, or code
    if containsPossibleInjection(promptSafe):
        return None

    # All printable test
    if not isPrintable(promptSafe):
        return None

    return promptSafe

def sanitizeSource(source):
    """
    Validates the 'source' field. Returns source if valid, else None.
    """
    if source is None or not isinstance(source, str):
        return None

    source = source.strip()
    if not source or len(source) == 0:
        return None
    if not isPrintable(source):
        return None
    if len(source) > 256:
        return None

    return source

def main():
    """
    Main function to process input and output files with data validation/sanitization.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="path to the input JSON file")
    parser.add_argument("output_file", help="path to the output JSONLINES file")
    args = parser.parse_args()

    with open(args.input_file, "r") as inputFile:
        inputData = jsonlines.Reader(inputFile)

        with open(args.output_file, "w") as outputFile:
            outputData = jsonlines.Writer(outputFile)

            for jsonObj in inputData:
                # Defensive checks for required structure
                if ("text" not in jsonObj or
                    "metadata" not in jsonObj or
                    not isinstance(jsonObj["metadata"], dict) or
                    "source" not in jsonObj["metadata"]):
                    print("Skipping invalid object - missing required fields", file=sys.stderr)
                    continue

                prompt = jsonObj["text"]
                source = jsonObj["metadata"]["source"]

                promptSanitized = sanitizePrompt(prompt)
                if promptSanitized is None:
                    print("Skipping prompt due to invalid/toxic content.", file=sys.stderr)
                    continue

                sourceSanitized = sanitizeSource(source)
                if sourceSanitized is None:
                    print("Skipping object due to invalid 'source' field.", file=sys.stderr)
                    continue

                outputObj = {"00": promptSanitized, "source": sourceSanitized}
                outputData.write(outputObj)

if __name__ == "__main__":
    main()