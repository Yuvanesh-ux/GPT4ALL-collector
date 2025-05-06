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

# Constants for safety limits
MAX_PROMPT_LENGTH = 2000  # Maximum allowed prompt length in characters
MAX_RESPONSE_TOKENS = 512  # Safe maximum tokens to generate in a response

#store your api keys in a env file
load_dotenv()

class Conversation:
    def __init__(self, openai_api_keys: List[str]) -> None:
        self.openai_api_keys = openai_api_keys

    def truncate_prompt(self, prompt: str) -> str:
        """Truncate prompt to MAX_PROMPT_LENGTH characters."""
        if len(prompt) > MAX_PROMPT_LENGTH:
            logger.warning(f"Prompt length {len(prompt)} exceeds maximum {MAX_PROMPT_LENGTH} and will be truncated.")
            return prompt[:MAX_PROMPT_LENGTH]
        return prompt

    def get_responses(self, 
        all_prompts: List[str], 
        i: int, 
        output_path: str = "", 
        source: str = "",
        shard_size: int = 200,
    ):
        """A method that generates responses to a list of prompts using OpenAI's GPT-3.5-turbo model.

        Args:
            self: An instance of the class that this method belongs to.
            all_prompts (List[dict]): A list of prompts as dictionary objects to generate responses to. Each dictionary object should
            contain the 'prompt' key with a string value representing the prompt.
            i (int): An integer representing the starting index of the prompts to use in the all_prompts list.
            output_path (str, optional): The path to the file to write the generated responses to. Defaults to an empty string.
            source (str, optional): A string representing the source of the prompts. Defaults to an empty string.
            shard_size (int, optional): An integer representing the number of prompts to generate responses for in each iteration. Defaults to 200.

        Raises:
            Any exceptions thrown by the OpenAIChat model.

        """
        prompts = all_prompts[i : i + shard_size]
        model = OpenAIChat(
            model_name="gpt-3.5-turbo",
            openai_api_key=self.openai_api_keys[random.randint(0, len(self.openai_api_keys) - 1)],
            model_kwargs={"max_tokens": MAX_RESPONSE_TOKENS},
        )

        for prompt in tqdm(prompts):
            prompt = self.truncate_prompt(prompt)
            with jsonlines.open(output_path, mode="a") as writer:
                try:
                    turn_1 = model(prompt)
                    follow_up_prompt = "Write an insightful follow up question given the previous context: "
                    composed_prompt_q2 = follow_up_prompt + prompt + (turn_1 if turn_1 else "")
                    composed_prompt_q2 = self.truncate_prompt(composed_prompt_q2)
                    question_2 = model(composed_prompt_q2)
                    turn_2_context = prompt + (turn_1 if turn_1 else "")
                    composed_prompt_t2 = turn_2_context + (question_2 if question_2 else "")
                    composed_prompt_t2 = self.truncate_prompt(composed_prompt_t2)
                    turn_2 = model(composed_prompt_t2)
                    json_data = {
                        "00": prompt,
                        "01": turn_1,
                        "02": question_2,
                        "03": turn_2,
                        "source": source,
                    }
                    writer.write(json_data)
                except Exception:
                    logger.warning("something went wrong! next")
                    with jsonlines.open(f"{output_path}_fails.jsonl", mode="a") as writer:
                        writer.write(prompt)

    def conversation_collector(self, 
        all_prompts: List[str],
        output_path: str = "", 
        source: str = "",
        shard_size: int = 200,
        num_workers: int = 10
    ):
        """A method that generates responses to a list of prompts using OpenAI's GPT-3.5-turbo model.

        Args:
            self: An instance of the class that this method belongs to.
            all_prompts (List[str]): A list of prompts to generate responses to.
            i (int): An integer representing the starting index of the prompts to use in the all_prompts list.
            output_path (str, optional): The path to the file to write the generated responses to. Defaults to an empty string.
            source (str, optional): A string representing the source of the prompts. Defaults to an empty string.
            shard_size (int, optional): An integer representing the number of prompts to generate responses for in each iteration. Defaults to 200.
            num_workers (int, optional): An integer representing the number of worker processes to use. Defaults to 10.

        Raises:
            Any exceptions thrown by the OpenAIChat model.

        """
        logger.info("Generating Pairs")

        progress = tqdm(total=len(all_prompts) // shard_size)
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.get_responses, i=i, all_prompts=all_prompts, output_path=output_path, shard_size=shard_size, source=source) for i in range(0, len(all_prompts), shard_size)]

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
    else:
        num_of_keys = 25
        open_api_keys = []
        for i in range(1, num_of_keys + 1):
            key = os.environ.get(f'OPENAI_API_KEY{i}')
            if key:
                open_api_keys.append(key)
        if not open_api_keys:
            print("You need an api key!")
            exit()

    converse = Conversation(open_api_keys)

    documents = []
    with jsonlines.open(args.input_file, mode='r') as reader:
        for item in reader:
            prompt = item["00"]
            documents.append(prompt)
    
    converse.conversation_collector(all_prompts=documents, output_path=args.output_file)