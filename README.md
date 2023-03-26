
<h1 align="center"> prompt-scrape </h1>

<center>

<p> prompt-scrape allows you to mass scrape the ChatGPT API, allowing for input/output pairs in the millions to finetune your own model for conversational use, allowing the opensourcing of datasets and models alike (unlike what a certain inudstry leader is doing.) <p>

</center>


## Features

- Mass scrape the ChatGPT API
- Obtain input/output pairs in the millions
- Open source the datasets to create your amazing models!


## Installation

1. Clone the repository: `git clone https://github.com/yuvanesh-ux/prompt-scrape.git`
2. Install the required dependencies: `pip install -r requirements.txt`


## Usage

1. Navigate to the prompt-scrape directory.
2. Create your own file with input prompts. The input prompts should be saved in JSONL format.
3. Set the OpenAI API keys in your environment variables. You can also modify the openai_api_keys parameter in `Scraper()` to pass the keys as a list.
4. Run `python scrape.py`
5. The script will generate output JSONL files containing the prompt and response pairs, along with the model settings and source. The output files will be saved in the specified output path. Note that the output files will be appended if the path already exists.
6. You can modify the num_workers and shard_size parameters in `scrape()` to change the number of workers and the number of prompts processed per worker, respectively.

Note: You will need a ChatGPT API key(s) to use this tool. You can obtain a key from the OpenAI website.


## Contributing

Contributions are welcome! If you encounter any bugs or have suggestions for new features, please open an issue or submit a pull request.


## License

This project is licensed under the MIT License. See the LICENSE file for more information.
