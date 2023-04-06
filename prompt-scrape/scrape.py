import concurrent
import concurrent.futures
import os
import random
from typing import List

import jsonlines
from dotenv import load_dotenv
from langchain.llms import OpenAIChat
from loguru import logger
from tqdm import tqdm

load_dotenv()

class Scraper:
    def __init__(self, openai_api_keys: List[str]):
        self.openai_api_keys = openai_api_keys

    def get_responses(
        self, 
        all_prompts: List[dict], 
        i: int, 
        shard_size: int,
        model_settings: dict = {"max_tokens": -1},
        output_path: str = '',
        source: str = ''
    ):
        """A method that generates responses to a list of prompts using OpenAI's GPT-3.5-turbo model and writes the output to a file.

        Args:
            all_prompts (List[dict]): A list of prompts as dictionary objects to generate responses to. Each dictionary object should
            contain the 'text' key with a string value representing the prompt.
            i (int): An integer representing the starting index of the prompts to use in the all_prompts list.
            shard_size (int): An integer representing the number of prompts to generate responses for in each iteration.
            model_settings (dict, optional): A dictionary of settings to pass to the OpenAIChat model. Defaults to {"max_tokens": -1}.
            output_path (str, optional): The path to the directory to write the generated responses to. Defaults to an empty string.
            source (str, optional): A string representing the source of the prompts. Defaults to an empty string.

        Raises:
            Any exceptions thrown by the OpenAIChat model or jsonlines module.

        """
        prompts = all_prompts[i : i + shard_size]
        model = OpenAIChat(
            model_name="gpt-3.5-turbo",
            openai_api_key=self.openai_api_keys[random.randint(0, len(self.openai_api_keys) - 1)],
            model_kwargs={"max_tokens": -1}, # -1 specifies we want the maximum number of tokens that can be generated
        )
        for prompt in tqdm(prompts):
            output = model(prompt)
            with jsonlines.open(output_path, mode="a") as writer:
                try:
                    json_data = {"turn_0_question": prompt, "turn_0_answer": output, "model_settings": model_settings, "source": source}
                    writer.write(json_data)
                except (KeyboardInterrupt, ValueError, IndexError):
                    logger.warning("Something went wrong with this prompt! Skipping to next one")
                    with jsonlines.open(os.path.join(output_path, "fails.jsonl"), mode="a") as writer:
                        writer.write(prompt)

    def collector(self, 
        all_prompts: List[dict],
        num_workers: int = 10,
        shard_size: int = 200,
        output_path: str = '',
        source: str = ''
    ):
        logger.info("Generating Pairs")

        progress = tqdm(total=len(all_prompts) // shard_size)
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.get_responses, i=i, all_prompts=all_prompts, shard_size=shard_size, output_path=output_path, source=source) for i in range(0, len(all_prompts), shard_size)]

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.exception(f"Error processing prompt: {e}")

                progress.update(1)

if __name__ == "__main__":
    scraper = Scraper([os.environ[f'OPENAI_API_KEY{i}'] for i in range(1, 26)])

    documents = []
    with jsonlines.open('input_prompts/multi_sum_random_chunk_1.jsonl', mode='r') as reader:
        for item in reader:
            documents.append(item["turn_0_question"])
    
    cloud_path = '/mnt/efs/data/prompt-scrape-run-v2/output-multi_sum.jsonl'
    scraper.collector(all_prompts=documents[0:5], output_path='output_data/test_2.jsonl', source='unified_multi_sum')