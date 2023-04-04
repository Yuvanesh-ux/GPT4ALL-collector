from nomic import atlas
import jsonlines
import os
import re
from tqdm import tqdm
import numpy as np

def abstract_infill():
# abstract_infill
    file_name = 'abstract_infill'
    documents = []
    with jsonlines.open(os.path.join("input_prompts", f"unified_{file_name}.jsonl"), mode='r') as reader:
        for idx, item in enumerate(reader):
            json = {}
            try:
                background = re.search(r'Background:([\s\S]*?)<human>', item['text']).group(1).strip()
            except:
                background = ""

            # Extract each section between <human> and <bot>
            sections = re.findall(r'<human>:(.*?)<bot>:(.*?)\n', item['text'], re.DOTALL)

            json["background"] = ""
            if len(background) > 0:
                json["background"] = background
            for i, section in enumerate(sections):
                if i <= 5:
                    json[f"turn_{i}_question"] = section[0].strip()
                # print('Section', i+1, ':', section[0].strip())

            documents.append(json)

    with jsonlines.open("input_prompts/unified_abstract_infill_extracted.jsonl", mode='a') as writer:
        for doc in tqdm(documents):
            writer.write(doc)

    # # making every json have the same keys
    # for doc in documents:
    #     for i in range (6):
    #         if f"turn_{i}_question" not in doc:
    #             doc[f"turn_{i}_question"] = ""

    

    # # map explorer
    # atlas.map_text(
    #     data=documents,
    #     indexed_field='turn_0_question',
    #     name=f"{file_name} v3 prompts",
    #     description=f'{file_name} multiturn map exploration',
    #     organization_name="GPT4ALL"
    # )

def multi_news():
    file_name = 'multi_news'
    documents = []
    with jsonlines.open(os.path.join("input_prompts", f"unified_{file_name}.jsonl"), mode='r') as reader:
        for idx, item in enumerate(reader):
            if idx:
                json = {}
                # Extract each section between <human> and <bot>
                result = re.search(r'(?<=<human>:).*?(?=<bot>)', item["text"], re.DOTALL)

                json[f"turn_0_question"] = result.group(0).strip()
                documents.append(json)

    # making every json have the same keys
    # for doc in documents:
    #     for i in range(6):
    #         if f"turn_{i}_question" not in doc:
    #             doc[f"turn_{i}_question"] = ""

    with jsonlines.open("input_prompts/unified_multi_news_clean.jsonl", mode="a") as writer:
        for doc in tqdm(documents):
            writer.write(doc)
    
    exit()
    # map explorer
    atlas.map_text(
        data=documents,
        indexed_field='turn_0_question',
        name=f"{file_name} v4 prompts",
        description=f'{file_name} map exploration',
        # shard_size=100,
        # organization_name="GPT4ALL"
    )

def multi_sum():
    file_name = 'multi_sum'
    documents = []
    set = []
    # with jsonlines.open(os.path.join("input_prompts", f"unified_{file_name}.jsonl"), mode='r') as reader:
    #     for idx, item in tqdm(enumerate(reader)):
    #         set.append(item)

    # # np.random.seed(0)
    # # max_documents = 250_000
    # # subset_idxs = np.random.choice(len(set), size=max_documents, replace=False).tolist()
    # # set = [set[i] for i in subset_idxs]

    # with jsonlines.open("checkpoint.jsonl", mode='a') as writer:
    #     for idx, item in tqdm(enumerate(set)):
    #         json = {}
    #         # Extract each section between <human> and <bot>
    #         sections = re.findall(r'<human>:(.*?)<bot>', item['text'], flags=re.DOTALL)

    #         # print(len(sections))
    #         for idx, section in enumerate(sections):
    #             json[f"turn_{idx}_question"] = section.strip()
            
    #         writer.write(json)
    #         documents.append(json)

    with jsonlines.open("checkpoint.jsonl", mode="r") as reader:
        for item in tqdm(reader):
            documents.append(item)
    
    n = 100_000
    chunks = [documents[i:i + n] for i in range(0, len(documents), n)]
    for idx, chunk in enumerate(chunks):
        if idx == 0:
            project = atlas.map_text(
                        data=chunk,
                        indexed_field='turn_0_question',
                        name=f"{file_name} v9 prompts",
                        description=f'{file_name} map exploration',
                        # shard_size=100,
                        # organization_name="GPT4ALL"
                    )
        
        with project.wait_for_project_lock():
            project.add_text(
                data=chunk
            )
    
def hello_simple_hc3():
    documents =[]
    file_name = 'all'
    with jsonlines.open(os.path.join("input_prompts", f"{file_name}.jsonl"), mode='r') as reader:
        for idx, item in enumerate(reader):
            if idx:
                json = {}
                try:
                    # Extract each section between <human> and <bot>
                    json["turn_0_question"] = item["question"]
                    json["turn_0_answer"] = item["chatgpt_answers"][0]
                    documents.append(json)
                except:
                    print("failed")

    atlas.map_text(
        data=documents,
        indexed_field='turn_0_answer',
        name="hello-simple/hc3 v1",
    )

hello_simple_hc3()