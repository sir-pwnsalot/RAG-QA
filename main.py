import os
import logging
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# --- 1. SETUP & CONFIGURATION ---

# Load environment variables from .env file
load_dotenv()

# Configure basic logging to provide feedback
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration from environment variables
PERSIST_DIRECTORY = os.getenv("PERSIST_DIRECTORY")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
if not EMBEDDING_MODEL_NAME:
    logging.error("The EMBEDDING_MODEL_NAME environment variable is not set. Please set it in your .env file.")
    exit(1)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

# --- 2. CORE RAG COMPONENTS ---

def create_rag_chain(retriever, llm, prompt):
    """Creates the RAG chain using the specified retriever, LLM, and prompt."""
    try:
        # Create a RetrievalQA chain, which is a standard way to build Q&A systems
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # "stuff" means all retrieved text is stuffed into the prompt
            retriever=retriever,
            return_source_documents=True,  # This allows us to see which documents were used
            chain_type_kwargs={"prompt": prompt}
        )
        return qa_chain
    except Exception as e:
        logging.error(f"Failed to create RAG chain: {e}")
        return None

# --- 3. MAIN APPLICATION ---

def main():
    """Main function to run the interactive Q&A application."""
    logging.info("Starting the Q&A application.")

    # Initialize embeddings and load the vector store from disk
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        
        # Check if the database directory exists before trying to load it
        if not os.path.isdir(PERSIST_DIRECTORY):
            logging.error(f"Database directory not found at '{PERSIST_DIRECTORY}'. Please run ingest.py first.")
            return

        vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
        retriever = vectordb.as_retriever() # Create a retriever from the vector store
        logging.info("Vector store loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load vector store: {e}")
        return

    # Initialize the Ollama LLM
    try:
        llm = Ollama(model=OLLAMA_MODEL)
        logging.info(f"Ollama model '{OLLAMA_MODEL}' loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load Ollama model. Is Ollama running? Error: {e}")
        return

    # Define the prompt template to guide the LLM's answers
    prompt_template = """
    Use the following pieces of context to answer the question at the end. 
    If you don't know the answer from the context, just say that you don't know. Do not try to make up an answer.
    Keep the answer concise and helpful.

    Context: {context}

    Question: {question}

    Helpful Answer:
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Create the RAG chain
    qa_chain = create_rag_chain(retriever, llm, PROMPT)
    if not qa_chain:
        logging.error("Exiting due to RAG chain creation failure.")
        return

    # Start the interactive query loop
    print("\n--- Personal Note Q&A ---")
    print("Ask questions based on your documents. Type 'exit' or 'quit' to end.")
    
    try:
        while True:
            query = input("\nQuestion: ").strip()
            if query.lower() in ["exit", "quit"]:
                print("Exiting application. Goodbye!")
                break
            if not query:
                continue

            # Get the answer from the RAG chain
            logging.info(f"Processing query: '{query}'")
            result = qa_chain.invoke({"query": query})
            
            # Display the answer and its sources
            print("\n--- Answer ---")
            print(result["result"])
            print("\n--- Sources ---")
            for doc in result["source_documents"]:
                # Print the source file path from the document's metadata
                print(f"- {doc.metadata.get('source', 'Unknown source')}")

    except KeyboardInterrupt:
        print("\nApplication interrupted by user. Exiting.")
    except Exception as e:
        logging.error(f"An error occurred during the query loop: {e}")

if __name__ == "__main__":
    main()