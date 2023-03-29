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

openai_api_keys = [os.environ[f'OPENAI_API_KEY{i}'] for i in range(22, 26)]

shard_size = 200
num_workers = 10

class Conversation():
    def __init__(self) -> None:
        pass

    def get_responses(self, all_prompts: List[str], i: int, output_path: str = "", source: str = ""):
        prompts = all_prompts[i: i + shard_size]
        model = OpenAIChat(model_name="gpt-3.5-turbo", openai_api_key=openai_api_keys[random.randint(0,len(openai_api_keys)-1)], model_kwargs={"max_tokens": -1})

        for prompt in tqdm(prompts):
            with jsonlines.open(output_path, mode="a") as writer:
                try:
                    turn_1 = model(prompt)
                    follow_up_prompt = 'Write an insightful follow up question given the previous context: '
                    question_2 = model(follow_up_prompt + prompt + turn_1)
                    turn_2_context = prompt + turn_1
                    turn_2 = model(turn_2_context + question_2)
                    json_data = {"prompt" : prompt, "response_turn_1": turn_1, "prompt_turn_2": question_2, "response_turn_2": turn_2, "source": source}
                    writer.write(json_data)
                except:
                    logger.warning("something went wrong! next")
                    with jsonlines.open(f"{output_path}/fails.jsonl", mode="a") as writer:
                        writer.write(prompt)

    def scrape(self, all_prompts: List[str]):
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
    all_prompts = []
    with jsonlines.open("input_prompts/laion_leftovers", mode="r") as reader:
        for idx, datum in enumerate(reader):
            all_prompts.append(datum["prompt"])

    conversation = Conversation()
    conversation.scrape(all_prompts[0:10])

    