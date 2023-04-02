
<h1 align="center"> GPT4ALL-collector </h1>

<center>

<p> GPT4ALL-collector allows you to mass collect the ChatGPT API, allowing for input/output pairs in the millions to finetune your own model for conversational use, allowing the opensourcing of datasets and models alike (unlike what a certain inudstry leader is doing.) <p>

</center>


## Features

- Mass "collect" the ChatGPT API
- Obtain input/output pairs in the millions
- Open source the datasets to create your amazing models!


## Installation

1. Clone the repository: `git clone https://github.com/Yuvanesh-ux/GPT4ALL-collector.git`
2. Install the required dependencies: `pip install -r requirements.txt`


## Usage

1. Navigate to the prompt-scrape directory.
2. Create your own file with input prompts. Here is an example of an input file that will work out of the box with the scraper.py without modification:

```
{"prompt": "Can you write me a poem about kenneth fearing, aphrodite and jubal fearing in the style of KENNETH FEARING?", "source": "OIG - unified_poetry_instructions.jsonl"}
{"prompt": "Can you write me a poem about wallace stevens and alfred a. knopf?", "source": "OIG - unified_poetry_instructions.jsonl"}
{"prompt": "Can you write me a poem about time?", "source": "OIG - unified_poetry_instructions.jsonl"}
```

Other file formats will require either conversion to this format or modifications to scraper.py to accomodate your own custom format.

3. Run `python scrape.py -k <OPENAI_API_KEY> /path/to/your/input_file.jsonl /path/to/your/output_file.jsonl`
4. You can also set your OpenAI API keys to OPENAI_API_KEY environment variable.
5. The script will generate output a JSON file containing the prompt and response pairs, along with the model settings and source. Note that the output files will be appended if the path already exists.
6. You can modify the num_workers and shard_size parameters in `scrape()` to change the number of workers and the number of prompts processed per worker, respectively.

Note: You will need a ChatGPT API key(s) to use this tool. You can obtain a key from the OpenAI website.


## Contributing

Contributions are welcome! If you encounter any bugs or have suggestions for new features, please open an issue or submit a pull request.


## License

This project is licensed under the MIT License. See the LICENSE file for more information.
