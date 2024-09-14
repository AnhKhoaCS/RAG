from typing import Union
from langchain.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

class VectorDB:
    def __init__(self,
                document = None,
                vector_db: Union[Chroma, FAISS] = Chroma,
                embedding = HuggingFaceBgeEmbeddings(),
                ) -> None:
        self.vector_db = vector_db
        self.embedding = embedding
        self.db = self._builf_db(documents)

    def _build_db(self, documents):
        db = self.vector_db.from_documents(documents=documents, embedding=self.embedding)
        return db

    def get_retriever(self,
                      search_type: str = "similarity",
                      search_kwargs: dict = {"k" : 10}
                      ):
        retriever = self.db.as_retriever(search_type = search_type,
                                        search_kwargs = search_kwargs)
        return retrieve