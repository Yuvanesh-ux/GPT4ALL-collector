import argparse
import concurrent
import concurrent.futures
import os
import random
from typing import List
import queue
import threading

threads_to_start = 10 # or choose how many you want
my_queue = queue.Queue()


import jsonlines
import json
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
        source: str = '',
        progress = None
    ):
        """A method that generates responses to a list of prompts using OpenAI's GPT-3.5-turbo model and writes the output to a file.

        Args:
            all_prompts (List[dict]): A list of prompts as dictionary objects to generate responses to. Each dictionary object should
            contain the 'prompt' key with a string value representing the prompt.
            i (int): An integer representing the starting index of the prompts to use in the all_prompts list.
            shard_size (int): An integer representing the number of prompts to generate responses for in each iteration.
            model_settings (dict, optional): A dictionary of settings to pass to the OpenAIChat model. Defaults to {"max_tokens": -1}.
            output_path (str, optional): The path to the directory to write the generated responses to. Defaults to an empty string.
            source (str, optional): A string representing the source of the prompts. Defaults to an empty string.

        Raises:
            Any exceptions thrown by the OpenAIChat model or jsonlines module.

        """
        prompts = [d['prompt'] for d in all_prompts[i : i + shard_size]]
        model = OpenAIChat(
            model_name="gpt-3.5-turbo",
            openai_api_key=self.openai_api_keys[random.randint(0, len(self.openai_api_keys) - 1)],
            model_kwargs={"max_tokens": -1}, # -1 specifies we want the maximum number of tokens that can be generated
        )
        for prompt in prompts:
            output = model(prompt)
            with jsonlines.open(output_path, mode="a") as writer:
                try:
                    json_data = {"prompt": prompt, "response": output, "model_settings": model_settings, "source": source}
                    writer.write(json_data)
                except (KeyboardInterrupt, ValueError, IndexError):
                    logger.warning("Something went wrong with this prompt! Skipping to next one")
                    with jsonlines.open(output_path + "_fails.jsonl", mode="a") as writer:
                        writer.write(prompt)
            progress.update(1)
    def worker(self,progress):
        while not my_queue.empty():
            d = my_queue.get()
            self.get_responses(i=d[0], all_prompts=d[1], shard_size=d[2], output_path=d[3], source=d[4],progress=progress)
            my_queue.task_done()

    def collector(self, 
        all_prompts: List[dict],
        num_workers: int = 10,
        shard_size: int = 200,
        output_path: str = '',
        source: str = ''
    ):
        logger.info("Generating Pairs")

        progress = tqdm(total=len(all_prompts))        for i in range(0, len(all_prompts), shard_size):
               my_queue.put([i, all_prompts, shard_size, output_path, source])
        for i in range(threads_to_start):
                t = threading.Thread(target=self.worker, daemon=True, args=[progress]) 
                t.start()
        my_queue.join()
         
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Input file name")
    parser.add_argument("output_file", help="Output file name")
    parser.add_argument("-k", "--openai_api_key", help="OpenAI API key")
    args = parser.parse_args()

    # Get the OpenAI API key
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    # Create a Scraper instance
    scraper = Scraper(openai_api_keys=[openai_api_key])

    # Read the input prompts
    all_data = []
    with open(args.input_file, mode="r", encoding="latin-1") as f:
        data = f.read()
        
        for idx, line in enumerate(data.split('\n')):
            try:
                datum = json.loads(line.strip())
                prompt = datum['prompt']
                all_data.append({"prompt": prompt})
            except:
                print(f"Line {idx + 1} failed")

    # Scrape the prompts
    scraper.collector(
        all_prompts=all_data,
        output_path=args.output_file
    )
