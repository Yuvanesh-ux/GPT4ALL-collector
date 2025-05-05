import argparse
import jsonlines
import sys
from nomic import atlas

# Maximum number of items to process from the input file
MAX_ITEMS = 100000

# create an ArgumentParser object and define the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("input_file", help="path to the input json file")
args = parser.parse_args()

# load the data from the input file using jsonlines, with item limit
data = []
with jsonlines.open(args.input_file, "r") as input_file:
    for i, obj in enumerate(input_file):
        if i >= MAX_ITEMS:
            print(f"Error: File contains more than {MAX_ITEMS} items. Refusing to process further to avoid excessive memory usage.", file=sys.stderr)
            sys.exit(1)
        if 'source' not in obj:
            obj['source'] = args.input_file
        data.append(obj)

print(f"processing {len(data)} items")

# index the prompt field using atlas.map_text()
project = atlas.map_text(data=data,
                         indexed_field="00",
                         name=args.input_file,
                         colorable_fields=["source"]
                        )