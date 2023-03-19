from nomic import atlas
import jsonlines

documents = []
with jsonlines.open("output_1000_2000.jsonl", mode="r") as reader:
    for datum in reader:
        del datum["model_settings"]
        if any(isinstance(v, dict) for v in datum.values()) or datum["response"] is None:
            continue
        # data_to_append = {}
        # data_to_append["prompt"] = datum["prompt"]
        # data_to_append["response"] = datum["response"]
        documents.append(datum)



project = atlas.map_text(data=documents,
                          indexed_field='response',
                          name='ChatGPT 1k Output LAION OIB V2',
                          description='ChatGPT 1k Output (Max Token Update)'
                        )