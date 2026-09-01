import os
import chromadb
import ollama

# 1. Initialize local ChromaDB (Vector Store)
db_client = chromadb.PersistentClient(path="./chroma_db")
collection = db_client.get_or_create_collection(name="myfinergy_kb")

# 2. Populate Database if empty
if collection.count() == 0:
    kb_path = "data/knowledge_base.txt"
    if os.path.exists(kb_path):
        with open(kb_path, "r", encoding="utf-8") as f:
            text_data = f.read()

        # Split text into sections using double newlines
        chunks = [chunk.strip() for chunk in text_data.split("\n\n") if chunk.strip()]
        
        # Add chunks into vector store
        for idx, chunk in enumerate(chunks):
            collection.add(
                documents=[chunk],
                ids=[f"chunk_{idx}"]
            )
        print("Indexed Knowledge Base into ChromaDB successfully!")
    else:
        print(f"Error: Could not find {kb_path}. Run create_kb.py first.")

SYSTEM_PROMPT = """
You are the official AI Assistant for MyFinergy (myfinergy.com).
Your job is to answer customer support and product questions accurately using ONLY the provided company context below.

STRICT RULES:
1. MyFinergy is a FINANCIAL DIAGNOSTIC TOOL for advisors, NOT a financial planning/execution platform.
2. Never offer stock, mutual fund, or specific investment recommendations to clients.
3. Keep answers clear, direct, and professional.
4. If you do not know the answer based on the context, say: "I don't have that specific detail. Please reach out to support@myfinergy.com or request a demo on www.myfinergy.com."
"""

def chat():
    print("\n" + "="*50)
    print("      MyFinergy AI Support Bot Ready (Offline Mode)")
    print("      Type 'exit' or 'quit' to end the chat.")
    print("="*50 + "\n")

    while True:
        user_query = input("You: ").strip()
        if not user_query:
            continue
        if user_query.lower() in ["exit", "quit"]:
            print("Bot: Goodbye!")
            break

        # Retrieve relevant sections from local vector database
        results = collection.query(query_texts=[user_query], n_results=2)
        retrieved_docs = results.get("documents", [[]])[0]
        context = "\n---\n".join(retrieved_docs) if retrieved_docs else "No specific context found."

        # Send query + retrieved context to local Ollama model
        prompt = f"Company Context:\n{context}\n\nUser Question: {user_query}"
        
        try:
            response = ollama.chat(
                model="llama3.2",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ]
            )
            print(f"\nBot: {response['message']['content']}\n")
        except Exception as e:
            print(f"\nError interacting with Ollama: {e}")
            print("Ensure Ollama app is running in your Mac menu bar.\n")

if __name__ == "__main__":
    chat()