from langchain.prompts import PromptTemplate
from langchain.llms import OpenAIChat
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
import json
import jsonlines
from tqdm import tqdm
from loguru import logger
from timeit import default_timer as timer
# from concurrent.futures import ProcessPoolExecutor
# import itertools


# model = OpenAIChat(model_name="gpt-3.5-turbo", openai_api_key="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")

class Parse_Model(BaseModel):
    prompt: str = Field(description="prompt supplied to model")
    response: str = Field(description="response to supplied prompt")

parser = PydanticOutputParser(pydantic_object=Parse_Model)

prompt_template = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=['query'],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

class Scraper():
    def __init__(self) -> None:
        pass

    def scrape(self, all_prompts: List[dict], openai_api_key: str):

        logger.warning("Scraping Starts")
        def generate_pairs(prompts, open_api_key):
            logger.warning("Generating Pairs")
            model = OpenAIChat(model_name="gpt-3.5-turbo", openai_api_key=open_api_key)

            pairs = []
            for prompt in tqdm(prompts[:1000]):
                _input = prompt_template.format_prompt(query=prompt)

                output = model(_input.to_string())

                # print(parser.parse(output))
                # try:
                #     data = json.loads(parser.parse(output).json())
                #     pairs.append(data)
                # except:
                #     logger.warning("Pydantic Validation Failed!")

                try:
                    pairs.append(json.loads(output))
                except:
                    logger.warning("Something went wrong with this prompt! Skipping to next one")
        
            return pairs


        results = generate_pairs(prompts=all_prompts, open_api_key=openai_api_key)

        return results



if __name__ == "__main__":
    start = timer()

    scraper = Scraper()

    all_data = []

    with jsonlines.open('all_data.jsonl', mode='r') as reader:
        for idx, datum in enumerate(reader):
            all_data.append(datum)

    openai_api_key = "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"

    results = scraper.scrape(all_prompts=all_data, openai_api_key=openai_api_key)
    
    logger.warning("Writing to disk")
    with jsonlines.open("output.jsonl", mode="a") as writer:
        for result in tqdm(results): 
            writer.write(result)

    end = timer()
    print(end - start)