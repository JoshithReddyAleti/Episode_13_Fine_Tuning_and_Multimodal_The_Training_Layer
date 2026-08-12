'''Illustrate a multimodal RAG query flow.'''

STAGES = [
    "1. User query (text or image or both)",
    "2. Query embedding (multimodal encoder)",
    "3. Retrieval fanned to modalities:",
    "     - text embeddings (BGE-style)",
    "     - image embeddings (SigLIP)",
    "     - video frame embeddings (ColPali-style)",
    "     - audio transcript embeddings",
    "4. Reciprocal rank fusion across modalities",
    "5. Reranking (optional, per modality)",
    "6. Context assembly (top-K items, mixed modalities)",
    "7. Generation:",
    "     - VLM if images present",
    "     - LLM otherwise",
    "8. Answer + citations back to source",
]

if __name__ == "__main__":
    print("Multimodal RAG query flow:")
    for s in STAGES:
        print(s)
