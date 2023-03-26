import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="your_project_name",
    version="0.1.0",
    author="Yuvanesh Anand",
    description="A semi scalabale system of mass scraping the chatgpt API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yuvanesh-ux/prompt-scrape",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "python-dotenv",
        "langchain",
        "openai",
        "tqdm",
        "jsonlines"
    ],
    python_requires='>=3.6',
)