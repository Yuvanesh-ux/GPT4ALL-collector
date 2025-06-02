# config.py

"""
Configuration constants for LLM scraping
"""

# Maximum prompt length to accept (sanity check).
MAX_PROMPT_LENGTH = 4096  # LLM API safe upper bound

# Maximum number of tokens to generate in response (prevents abuse).
MAX_LLM_TOKENS = 1024  # You may adjust this based on your LLM quota.


# scrape.py

import argparse
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

from config import MAX_PROMPT_LENGTH, MAX_LLM_TOKENS

load_dotenv()

class Scraper:
    def __init__(self, openai_api_keys: List[str]):
        self.openai_api_keys = openai_api_keys

    def sanitize_prompt(self, prompt) -> str:
        """
        Sanitize an input prompt to prevent adversarial manipulation.
        
        Args:
            prompt: The prompt to sanitize.
            
        Returns:
            str: The sanitized prompt.
            
        Raises:
            ValueError: If the prompt is invalid.
        """
        # Check if the prompt is a string
        if not isinstance(prompt, str):
            raise ValueError("Prompt must be a string.")
        
        # Check if the prompt is empty
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")
        
        # Remove potential control characters, which could be used for manipulation
        prompt = ''.join(char for char in prompt if ord(char) >= 32 or char in '\n\t')
        
        # Limit the length of the prompt to prevent excessive large inputs
        max_prompt_length = MAX_PROMPT_LENGTH
        if len(prompt) > max_prompt_length:
            prompt = prompt[:max_prompt_length]
            
        return prompt

    def enforce_token_limits(self, tokens_requested):
        """
        Clamp the requested max_tokens to an allowed upper bound.

        Args:
            tokens_requested (int): Desired max token count.

        Returns:
            int: Authorised value, not exceeding MAX_LLM_TOKENS.
        """
        if not isinstance(tokens_requested, int):
            return MAX_LLM_TOKENS
        if tokens_requested < 1 or tokens_requested > MAX_LLM_TOKENS:
            return MAX_LLM_TOKENS
        return tokens_requested

    def get_responses(
        self, 
        all_prompts: List[dict], 
        i: int, 
        shard_size: int,
        model_settings: dict = None,
        output_path: str = '',
        source: str = ''
    ):
        """A method that generates responses to a list of prompts using OpenAI's GPT-3.5-turbo model and writes the output to a file.

        Args:
            all_prompts (List[dict]): A list of prompts as dictionary objects to generate responses to. Each dictionary object should
            contain the 'text' key with a string value representing the prompt.
            i (int): An integer representing the starting index of the prompts to use in the all_prompts list.
            shard_size (int): An integer representing the number of prompts to generate responses for in each iteration.
            model_settings (dict, optional): A dictionary of settings to pass to the OpenAIChat model.
            output_path (str, optional): The path to the directory to write the generated responses to. Defaults to an empty string.
            source (str, optional): A string representing the source of the prompts. Defaults to an empty string.

        Raises:
            Any exceptions thrown by the OpenAIChat model or jsonlines module.

        """
        prompts = all_prompts[i : i + shard_size]
        # Default model_settings if none provided
        if model_settings is None:
            model_settings = {"max_tokens": MAX_LLM_TOKENS}
        else:
            # Sanitize and enforce token caps
            if "max_tokens" not in model_settings or not isinstance(model_settings["max_tokens"], int):
                model_settings["max_tokens"] = MAX_LLM_TOKENS
            else:
                model_settings["max_tokens"] = self.enforce_token_limits(model_settings["max_tokens"])

        model = OpenAIChat(
            model_name="gpt-3.5-turbo",
            openai_api_key=self.openai_api_keys[
                random.randint(0, len(self.openai_api_keys) - 1)
            ],
            model_kwargs={
                "max_tokens": model_settings.get("max_tokens", MAX_LLM_TOKENS)
            },  # Replaces dangerous -1 with bounded value.
        )
        for prompt in tqdm(prompts):
            output = model(prompt)
            with jsonlines.open(output_path, mode="a") as writer:
                try:
                    json_data = {
                        "00": prompt,
                        "01": output,
                        "model_settings": model_settings,
                        "source": source,
                        "00_len": len(prompt),
                        "01_len": len(output)
                    }
                    writer.write(json_data)
                except (KeyboardInterrupt, ValueError, IndexError):
                    logger.warning("Something went wrong with this prompt! Skipping to next one")
                    with jsonlines.open(
                        os.path.join(output_path, "fails.jsonl"), mode="a"
                    ) as fail_writer:
                        fail_writer.write(prompt)

    def collector(self, 
        all_prompts: List[dict],
        num_workers: int = 10,
        shard_size: int = 200,
        output_path: str = '',
        source: str = ''
    ):
        """
        Collects responses concurrently using multiple processes.

        Args:
            all_prompts (List[dict]): List of prompts.
            num_workers (int): Number of worker processes.
            shard_size (int): Number of prompts per shard.
            output_path (str): Output file path.
            source (str): Source label.
        """
        logger.info("Generating Pairs")

        progress = tqdm(total=len(all_prompts) // shard_size)
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            # All ProcessPool workers will use token limits enforced in get_responses
            futures = [
                executor.submit(
                    self.get_responses,
                    i=i,
                    all_prompts=all_prompts,
                    shard_size=shard_size,
                    output_path=output_path,
                    source=source
                )
                for i in range(0, len(all_prompts), shard_size)
            ]

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.exception(f"Error processing prompt: {e}")

                progress.update(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="file path of the input file")
    parser.add_argument("output_file", help="file path of the output file")
    parser.add_argument("-k", "--openai_api_key", help="OpenAI API key")
    args = parser.parse_args()

    if args.openai_api_key:
        open_api_keys = [args.openai_api_key]
    elif os.environ.get("OPENAI_API_KEY1"):
        num_of_keys = 25
        open_api_keys = [os.environ[f'OPENAI_API_KEY{i}'] for i in range(1, num_of_keys + 1)]
    else:
        print("You need an api key!")
        exit()

    scraper = Scraper(open_api_keys)

    documents = []
    with jsonlines.open(args.input_file, mode='r') as reader:
        for item in reader:
            try:
                if "00" not in item:
                    logger.warning("Missing prompt key in item. Skipping.")
                    continue
                
                raw_prompt = item["00"]
                sanitized_prompt = scraper.sanitize_prompt(raw_prompt)
                documents.append(sanitized_prompt)
            except ValueError as e:
                logger.warning(f"Invalid prompt: {str(e)}. Skipping.")
            except Exception as e:
                logger.exception(f"Error processing prompt: {str(e)}. Skipping.")
    
    scraper.collector(all_prompts=documents, output_path=args.output_file)