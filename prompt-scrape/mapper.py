from nomic import atlas
import jsonlines

documents = []
for item in ["output_data/output_1000_2000.jsonl", "output_data/output_2000_3000.jsonl", "output_data/output_3000_4000.jsonl"]:
    with jsonlines.open(f"{item}", mode="r") as reader:
        for datum in reader:
            del datum["model_settings"]
            if any(isinstance(v, dict) for v in datum.values()) or datum["response"] is None:
                continue

            documents.append(datum)



project = atlas.map_text(data=documents,
                          indexed_field='response',
                          name='ChatGPT 3k Output LAION OIB V2',
                          description='ChatGPT 3k Output (Max Token Update)'
                        )