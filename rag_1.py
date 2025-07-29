from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI

import os
from dotenv import load_dotenv


load_dotenv()
from pathlib import Path
api_key=os.getenv("API_KEY")
# api_key=os.getenv("API_KEY")
client = OpenAI(api_key=api_key)

pdf_path = Path(__file__).parent/ "lokal_negative.pdf"

loader = PyPDFLoader(file_path = pdf_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200,
)

split_docs = text_splitter.split_documents(documents=docs)

# creating a function to embedding the model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=api_key,
)


## if it run first time the collection is not in the DB so it throws error
vector_store = QdrantVectorStore.from_existing_collection(  
    url="http://localhost:6333",
    collection_name="lokal_report",
    embedding=embeddings
)

#injecting each chunk's vector embedding to vector store
# It is done first time only
# vector_store = QdrantVectorStore.from_documents(
#     documents=[],
#     url="http://localhost:6333",
#     collection_name="lokal_report",
#     embedding=embeddings
# )

# vector_store.add_documents(documents=split_docs)
# print("Injection done")


# Retrival of 

retriever = QdrantVectorStore.from_existing_collection(  
    url="http://localhost:6333",
    collection_name="lokal_report",
    embedding=embeddings
)

# relevent_chunks = retriever.similarity_search(
#     query=query
# )


# print("Relevent chunks",relevent_chunks)



# print("length of split docs:",len(split_docs))
# print("length of  docs:",len(docs))
while True:
    query = input("\nEnter your question > : ")
    if query.lower() == "exit":
        break
    relevant_chunks = retriever.similarity_search(query=query)

    # print("Chunks Found:", len(relevant_chunks))
    if not relevant_chunks:
        print("No relevant context found. Try another query.")
        continue

    context = "\n\n".join([doc.page_content for doc in relevant_chunks])
    SYSTEM_PROMPT = f"""
        You are an Helpful AI Assistant who responds base of the available context.

        Context:
        {context}
        """

    response = client.chat.completions.create(
        model="gpt-4o",
        # temperature=0.3,
        # max_tokens=300,
        messages=[
             {"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": query} 
        ]
    )
    answer = response.choices[0].message.content
    print("\n Answer:\n", answer)
