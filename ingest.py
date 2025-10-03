import os
import logging
import json
from datetime import datetime
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format= '%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ingestion.log"),
        logging.StreamHandler()
    ]
)


SOURCE_DIRECTORY = os.getenv("SOURCE_DIRECTORY", "Documents")
PERSIST_DIRECTORY = os.getenv("PERSIST_DIRECTORY", "db")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
PROCESSED_FILES_JSON = os.getenv("PROCESSED_FILES_JSON", "processed_files.json")

def load_processed_files(filepath):
    """Loads the dictionary of processed files and their modification times."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
            # If the file doesn't exist or is empty/corrupt, start fresh
            return {}
    
def save_processed_files(filepath, processed_files):
    """Saves the dictionary of processed files and their modification times."""
    try:
        with open(filepath, 'w') as f:
            json.dump(processed_files, f, indent=4)
    except IOError as e:
        logging.error(f"IOError while saving processed files to {filepath}: {e}")
    except Exception as e:
        logging.error(f"Failed to save processed files to {filepath}: {e}")

def should_process_file(filepath, processed_files):
    """Determines if a file should be processed or not"""
    try:
        cm_time = os.path.getmtime(filepath)
    except OSError:
        return False

    if filepath not in processed_files:
        return True
    if processed_files[filepath] < cm_time:
        return True

    return False

def load_new_documents(source_path, processed_files):
    """Loads only new or modified markdown documents."""
    new_docs = []
    files_to_update_log = {}

    logging.info(f"Scanning directory: {source_path}")
    for root, _, files in os.walk(source_path):
        for filename in files:
            if filename.endswith(".md"):
                filepath = os.path.join(root, filename)
                if should_process_file(filepath, processed_files):
                    try:
                        logging.info(f"Loading new/modified file: {filepath}")
                        loader = UnstructuredMarkdownLoader(filepath)
                        new_docs.extend(loader.load())
                        files_to_update_log[filepath] = os.path.getmtime(filepath)
                    except Exception as e:
                        logging.error(f"Failed to load file {filepath}: {e}")
    
    return new_docs, files_to_update_log

def split_documents(documents):
    """Splits documents into smaller chunks."""
    logging.info(f"Splitting {len(documents)} document(s) into chunks.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    logging.info(f"Created {len(texts)} new chunks.")
    return texts

def update_vector_store(texts, embeddings_model, db_path):
    """Creates or updates the vector store with new documents."""
    if not texts:
        logging.info("No new texts to add to the vector store.")
        return

    try:
        logging.info("Initializing vector store...")
        vectordb = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings_model
        )
        logging.info(f"Adding {len(texts)} new chunks to the vector store.")
        vectordb.add_documents(texts)
        logging.info("Successfully updated the vector store.")
    except Exception as e:
        logging.error(f"Failed to update vector store: {e}")

# --- 4. MAIN ORCHESTRATOR ---

def main():
    """Main function to run the ingestion pipeline."""
    logging.info("--- Starting ingestion process ---")
    
    # Load the log of already processed files
    processed_files = load_processed_files(PROCESSED_FILES_JSON)
    
    # Load only new or updated documents
    new_documents, files_to_update = load_new_documents(SOURCE_DIRECTORY, processed_files)

    if not new_documents:
        logging.info("No new or modified documents to process. Exiting.")
        return
        
    # Split the new documents into chunks
    chunks = split_documents(new_documents)
    
    # Initialize the embedding model
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    # Update the vector store with the new chunks
    update_vector_store(chunks, embeddings, PERSIST_DIRECTORY)
    
    # Update and save the log of processed files
    processed_files.update(files_to_update)
    save_processed_files(PROCESSED_FILES_JSON, processed_files)
    
    logging.info("--- Ingestion process completed successfully ---")

if __name__ == "__main__":
    main()