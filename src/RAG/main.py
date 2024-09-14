from pydantic import BaseModel, Field

from RAG.file_loader import Loader 
from RAG.vectorstore import VectorDB
from RAG.offine_rag import offine_rag

class InputQA(BaseModel):
    question: str = Field(..., title = "Question to ask the model")

class OutputQA(BaseModel):
    answer: str = Field(..., title = "Answer from the model")

def build_rag_chain(llm, data_dir, data_type):
    doc_loader = Loader(file_type = data_type).load_dir(data_dir, workers = 2)
    retriever = VectorDB(document = doc_loader)/get_retriever()
    rag_chain = office_rag(llm).get_chain(retriever)

    return rag_chain