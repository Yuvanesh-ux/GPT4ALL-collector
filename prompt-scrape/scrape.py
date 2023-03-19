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

class Scraper():
    def __init__(self) -> None:
        pass

    def scrape(self, all_prompts: List[dict], openai_api_key: str, model_settings: dict):
        logger.info("Generating Pairs")
        model = OpenAIChat(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, model_kwargs=model_settings)

        failed = []
        with jsonlines.open("output_data/output_2000_3000.jsonl", mode="a") as writer:
            for prompt in tqdm(all_prompts):
                _input = prompt_template.format_prompt(query=prompt)

                output = model(_input.to_string())

                try:
                    json_data = output_parser.parse(output)
                    json_data["model_settings"] = model_settings
                    writer.write(json_data)
                except (KeyboardInterrupt, ValueError, IndexError):
                    logger.warning("Something went wrong with this prompt! Skipping to next one")
                    failed.append(output)

        with jsonlines.open("output_data/failed/failed_outputs.jsonl", mode="w") as writer:
            for fail in failed:
                writer.write(fail)

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

    openai_api_key = os.environ['OPENAI_API_KEY']

    scraper.scrape(all_prompts=all_data[2037:3000], openai_api_key=openai_api_key, model_settings={"max_tokens": -1})