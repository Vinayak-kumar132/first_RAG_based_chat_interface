# PDF-Based Question Answering with LangChain, Qdrant, and OpenAI

This project implements a **PDF-based question-answering chatbot** that:
- Loads and splits a PDF into chunks,
- Embeds each chunk using OpenAI's `text-embedding-3-large` model,
- Stores and retrieves vector embeddings using **Qdrant vector database**,
- Uses `GPT-4o` for answering user queries based on similar document chunks.

## 🛠 Tech Stack

- [LangChain](https://github.com/langchain-ai/langchain): Used for document loading, text splitting, and embedding.
- [Qdrant](https://qdrant.tech/): Open-source vector database for storing and querying embeddings.
- [OpenAI](https://platform.openai.com/): Used for both embedding generation and GPT-based chat completion.
- [Python-dotenv](https://pypi.org/project/python-dotenv/): For loading API keys securely from `.env`.

---






## ⚙️ Setup Instructions

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```
```bash
pip install langchain langchain-community langchain-openai langchain-qdrant qdrant-client openai python-dotenv
```
```bash
docker run -p 6333:6333 -v qdrant_data:/qdrant/storage qdrant/qdrant
```
```env
API_KEY=your_openai_api_key
```

```Python
python main.py
```




