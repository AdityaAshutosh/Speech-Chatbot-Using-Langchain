# Q&A Chatbot using Langchain

A Q&A chatbot using <b>Langchain</b> framework utilizing <b>llama3</b> for RAG pipeline. It is given a speech data from Andrew Huberman's <a href="https://www.youtube.com/watch?v=WDv4AWk0J3U"> podcast clip </a>, which is converted into processed text and later passed through RAG pipeline leveraging Langchain framework. 

## Project Structure

- Pre-processing of speech data stored in `dataset` folder is done in `preprocessing.ipynb` file using <b> Assembly AI </b> API, used for speech-to-text conversion.
-`retrieval.ipynb` file loads the processed text, transforms it into chunks using `RecursiveTextSplitter` available in `langchain.text_splitter` and embeds using `Google Embeddings`.
- `FAISS` is used as vector database.
- Using `Stuff Documents Chain` functionality to pass components prompt and LLM to fetch relevant context from vector retrieval chain to retrieve information from vector DB using `Retrieval Chain`

## Setup and Installation

To set up and run the project, follow these steps:

1. Clone the repository to your local machine:

   ```
   git clone https://github.com/AdityaAshutosh/Speech-Chatbot-Using-Langchain
   ```

2. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root directory and add your Google API key.
   
4. Download <b>ollama </b> from this link: <a href="https://ollama.ai/download"> https://ollama.ai/download </a>   

5. Run all the cells in `retrieval.ipynb` in Jupyter Notebook and type your query when prompted to.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.
