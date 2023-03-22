from langchain.prompts import PromptTemplate
from langchain.llms import OpenAIChat
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from typing import List, Any
import json
import jsonlines
from tqdm import tqdm
from loguru import logger
from timeit import default_timer as timer
from dotenv import load_dotenv
import os
import concurrent
import concurrent.futures
import multiprocessing


load_dotenv()

response_schemas = [
    ResponseSchema(name="prompt", description="prompt supplied to model"),
    ResponseSchema(name="response", description="response to supplied prompt")
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

prompt_template = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=['query'],
    partial_variables={"format_instructions": format_instructions}
)

openai_api_keys = [os.environ[f'OPENAI_API_KEY{i}'] for i in range(1, 11)]

num_workers = 10
shard_size = 200

class Scraper():
    def __init__(self) -> None:
        pass
            
    def get_responses(self, all_prompts: List[dict], i: int, model_settings: dict = {"max_tokens": -1}):
            prompts = all_prompts[i: i + shard_size]
            model = OpenAIChat(model_name="gpt-3.5-turbo", openai_api_key=openai_api_keys[i%10], model_kwargs={"max_tokens": -1})
            for prompt in tqdm(prompts):
                _input = prompt_template.format_prompt(query=prompt)

                output = model(_input.to_string())
                with jsonlines.open("output_data/output_4000_x.jsonl", mode="a") as writer:
                    try:
                        json_data = output_parser.parse(output)
                        json_data["model_settings"] = model_settings
                        writer.write(json_data)
                    except (KeyboardInterrupt, ValueError, IndexError):
                        logger.warning("Something went wrong with this prompt! Skipping to next one")
                        with jsonlines.open("output_data/failed/fails.jsonl", mode="w") as writer:
                            writer.write(prompt)

    def scrape(self, all_prompts: List[dict]):
        logger.info("Generating Pairs")

        progress = tqdm(total=len(all_prompts) // shard_size)
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.get_responses, i=i, all_prompts=all_prompts) for i in range(0, len(all_prompts), shard_size)]

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.exception(f"Error processing prompt: {e}")
                    
                progress.update(1)

if __name__ == "__main__":
    scraper = Scraper()

    all_data = []

    with jsonlines.open('input_prompts/unified_chip2.jsonl', mode='r') as reader:
        for datum in reader:
            start = "<human>: "
            end = "<bot>:"
            text = datum["text"]

            # Extracting the prompt from the existing input/output pairs in the dataset
            result = text[text.index(start)+len(start):text.index(end)]

            all_data.append(result)

    scraper.scrape(all_prompts=all_data[35000:55000])