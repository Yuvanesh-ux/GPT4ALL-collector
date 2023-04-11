import argparse
import jsonlines
from nomic import atlas

# create an ArgumentParser object and define the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("input_file", help="path to the input json file")
args = parser.parse_args()

# load the data from the input file using jsonlines
data = []
with jsonlines.open(args.input_file, "r") as input_file:
    for obj in input_file:
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
