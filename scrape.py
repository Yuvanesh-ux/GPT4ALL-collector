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
import re

load_dotenv()

# Safe maximum tokens per LLM response to prevent resource exhaustion
MAX_MODEL_TOKENS = 512

class Scraper:
    def __init__(self, openaiApiKeys: List[str]):
        self.openaiApiKeys = openaiApiKeys

    def sanitizePrompt(self, prompt: str) -> str:
        """
        Sanitize an input prompt to prevent adversarial manipulation via prompt injection.
        Blocks common prompt injection patterns and removes hazardous tokens.

        Args:
            prompt: The prompt to sanitize.

        Returns:
            str: The sanitized prompt.

        Raises:
            ValueError: If the prompt is invalid or contains disallowed patterns.
        """
        if not isinstance(prompt, str):
            raise ValueError("Prompt must be a string.")

        # Remove leading/trailing whitespace
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Prompt cannot be empty.")

        # Check for common LLM prompt injection patterns & role tokens
        lower_prompt = prompt.lower()
        prompt_injection_patterns = [
            r"^(system:)",          # role prefix at beginning of prompt
            r"^(user:)",
            r"^(assistant:)",
            r"<\|.*?\|>",           # openai role token anywhere
            r"(\[system\])",
            r"(\[user\])",
            r"(\[assistant\])",
            r"\bignore all previous instructions\b",
            r"\bdisregard above\b",
            r"\breset the conversation\b",
            r"\bplease follow the next instructions\b",
            r"^#",                  # markdown/system shell start
        ]
        for pattern in prompt_injection_patterns:
            if re.search(pattern, lower_prompt, re.MULTILINE):
                raise ValueError("Unsafe prompt detected: potential prompt injection marker found.")

        # Remove control characters (ASCII < 32 except for \n, \t)
        prompt = ''.join(char for char in prompt if ord(char) >= 32 or char in '\n\t')
        maxPromptLength = 4096
        if len(prompt) > maxPromptLength:
            prompt = prompt[:maxPromptLength]

        # Remove OpenAI special role tokens appearing anywhere in the string
        prompt = re.sub(r"<\|\s*(system|user|assistant)\s*\|>", "", prompt, flags=re.IGNORECASE)
        prompt = re.sub(r"\[(system|user|assistant)\]", "", prompt, flags=re.IGNORECASE)

        # Check again for emptiness after cleaning
        if not prompt.strip():
            raise ValueError("Prompt is empty or unsafe after sanitization.")

        # Prevent double newline (role split) at beginning which could confuse LLM
        if re.match(r"^\s*(system:|assistant:|user:)", prompt, re.IGNORECASE):
            raise ValueError("Prompt starts with reserved role prefix.")

        return prompt

    def getResponses(
        self,
        allPrompts: List[str],
        i: int,
        shardSize: int,
        modelSettings: dict = None,
        outputPath: str = '',
        source: str = ''
    ):
        """
        A method that generates responses to a list of prompts using OpenAI's GPT-3.5-turbo model and writes the output to a file.

        Args:
            allPrompts (List[str]): A list of sanitized prompts as strings.
            i (int): An integer representing the starting index of the prompts to use in the allPrompts list.
            shardSize (int): An integer representing the number of prompts to generate responses for in each iteration.
            modelSettings (dict, optional): A dictionary of settings to pass to the OpenAIChat model. Defaults to {"max_tokens": MAX_MODEL_TOKENS}.
            outputPath (str, optional): The path to the directory to write the generated responses to. Defaults to an empty string.
            source (str, optional): A string representing the source of the prompts. Defaults to an empty string.

        Raises:
            Any exceptions thrown by the OpenAIChat model or jsonlines module.

        """
        prompts = allPrompts[i : i + shardSize]

        # Sanitize incoming model settings and enforce max tokens cap
        if modelSettings is None:
            modelSettings = {"max_tokens": MAX_MODEL_TOKENS}
        else:
            modelSettings = dict(modelSettings)
            maxTokens = modelSettings.get("max_tokens", MAX_MODEL_TOKENS)
            if not isinstance(maxTokens, int) or maxTokens <= 0 or maxTokens > MAX_MODEL_TOKENS:
                modelSettings["max_tokens"] = MAX_MODEL_TOKENS
            else:
                modelSettings["max_tokens"] = maxTokens

        model = OpenAIChat(
            model_name="gpt-3.5-turbo",
            openai_api_key=self.openaiApiKeys[random.randint(0, len(self.openaiApiKeys) - 1)],
            model_kwargs={"max_tokens": modelSettings["max_tokens"]},
        )
        for prompt in tqdm(prompts):
            output = model(prompt)
            with jsonlines.open(outputPath, mode="a") as writer:
                try:
                    jsonData = {
                        "00": prompt,
                        "01": output,
                        "model_settings": modelSettings,
                        "source": source,
                        "00_len": len(prompt),
                        "01_len": len(output)
                    }
                    writer.write(jsonData)
                except (KeyboardInterrupt, ValueError, IndexError):
                    logger.warning("Something went wrong with this prompt! Skipping to next one")
                    with jsonlines.open(os.path.join(outputPath, "fails.jsonl"), mode="a") as failWriter:
                        failWriter.write({"00": prompt})

    def collector(
        self,
        allPrompts: List[str],
        numWorkers: int = 10,
        shardSize: int = 200,
        outputPath: str = '',
        source: str = ''
    ):
        logger.info("Generating Pairs")

        progress = tqdm(total=len(allPrompts) // shardSize)
        with concurrent.futures.ProcessPoolExecutor(max_workers=numWorkers) as executor:
            futures = [
                executor.submit(
                    self.getResponses,
                    i=i,
                    allPrompts=allPrompts,
                    shardSize=shardSize,
                    outputPath=outputPath,
                    source=source
                ) for i in range(0, len(allPrompts), shardSize)
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
        openApiKeys = [args.openai_api_key]
    elif os.environ.get("OPENAI_API_KEY1"):
        numOfKeys = 25
        openApiKeys = [os.environ[f'OPENAI_API_KEY{i}'] for i in range(1, numOfKeys + 1)]
    else:
        print("You need an api key!")
        exit()

    scraper = Scraper(openApiKeys)

    documents = []
    with jsonlines.open(args.input_file, mode='r') as reader:
        for item in reader:
            try:
                if "00" not in item:
                    logger.warning("Missing prompt key in item. Skipping.")
                    continue

                rawPrompt = item["00"]
                sanitizedPrompt = scraper.sanitizePrompt(rawPrompt)
                documents.append(sanitizedPrompt)
            except ValueError as e:
                logger.warning(f"Invalid prompt: {str(e)}. Skipping.")
            except Exception as e:
                logger.exception(f"Error processing prompt: {str(e)}. Skipping.")

    scraper.collector(allPrompts=documents, outputPath=args.output_file)