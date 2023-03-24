from nomic import atlas
import jsonlines

documents = []
for item in ["input_prompts/all_prompts_mapping.jsonl"]:
    with jsonlines.open(f"{item}", mode="r") as reader:
        for datum in reader:
            try:
                # del datum["model_settings"]
                if any(isinstance(v, dict) for v in datum.values()) or datum["text"] is None:
                    continue
                documents.append(datum)
            except jsonlines.jsonlines.InvalidLineError:
                continue

# documents = []
# with jsonlines.open("input_prompts/all_data.jsonl") as reader:
#     with jsonlines.open("input_prompts/all_prompts_mapping.jsonl", mode='a') as writer:
#         for idx, datum in enumerate(reader):
#             if idx <= 280_000:
#                 json = {}
#                 json["text"] = datum
#                 writer.write(json)

# with jsonlines.open("input_prompts/unified_chip2.jsonl") as reader:
#     with jsonlines.open("input_prompts/all_prompts_mapping.jsonl", mode='a') as writer:
#         for idx, datum in enumerate(reader):
#             start = "<human>: "
#             end = "<bot>:"
#             text = datum["text"]

#             # Extracting the prompt from the existing input/output pairs in the dataset
#             result = text[text.index(start)+len(start):text.index(end)]
#             json["text"] = result
#             writer.write(json)


# exit()

project = atlas.map_text(data=documents,
                          indexed_field='text',
                          name='Prompt 500k ',
                          description='Prompt 500k'
                        )