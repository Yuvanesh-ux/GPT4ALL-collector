import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="GPT4ALL-collector",
    version="0.1.0",
    author="Yuvanesh Anand",
    description="A semi scalabale system of mass scraping the chatgpt API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Yuvanesh-ux/GPT4ALL-collector",
    install_requires=[
        "python-dotenv",
        "langchain",
        "openai",
        "tqdm",
        "jsonlines",
        "loguru"
    ],
    python_requires='>=3.6',
)