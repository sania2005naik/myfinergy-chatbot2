# database.py
import chromadb
from faq_data import MYFINERGY_FAQS

chroma_client = chromadb.PersistentClient(path="./chroma_db")

def populate_database():
    try:
        chroma_client.delete_collection(name="myfinergy_faqs_v5")
    except Exception:
        pass
        
    collection = chroma_client.create_collection(
        name="myfinergy_faqs_v5",
        metadata={"hnsw:space": "cosine"}
    )

    documents = []
    metadatas = []
    ids = []

    for item in MYFINERGY_FAQS:
        # Create a rich searchable string with question, keywords, and category context
        searchable_text = f"Question: {item['question']} | Category: {item['category']} | Keywords: {item['question'].lower()}"
        
        documents.append(searchable_text)
        metadatas.append({
            "category": item["category"], 
            "question": item["question"],
            "answer": item["answer"]
        })
        ids.append(str(item["id"]))

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print(f"Successfully re-indexed {len(documents)} FAQs into 'myfinergy_faqs_v5'!")

def search_faq(user_query: str, n_results: int = 1):
    collection = chroma_client.get_collection(name="myfinergy_faqs_v5")
    results = collection.query(
        query_texts=[user_query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    return results

if __name__ == "__main__":
    populate_database()