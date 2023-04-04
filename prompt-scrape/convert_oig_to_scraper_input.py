import argparse
import jsonlines

# create an ArgumentParser object and define the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("input_file", help="path to the input JSON file")
parser.add_argument("output_file", help="path to the output JSONLINES file")
args = parser.parse_args()

# open the input file and read the data
with open(args.input_file, "r") as input_file:
    input_data = jsonlines.Reader(input_file)

    # open the output file and write the output data as a JSONLINES file
    with open(args.output_file, "w") as output_file:
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
            output_obj = {"prompt": prompt, "source": source}
            output_data.write(output_obj)
