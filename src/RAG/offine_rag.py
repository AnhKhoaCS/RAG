import re
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class Str_OutputParser(StrOutputParser):
    def __init__(self) -> None:
        super().__init__()

    def parsel(self, text: str) -> str:
        return seft.extract_answer(text)


    def extract_answer(self, 
                    text_response: str,
                    pattern: str = r"Answer:\s*(.*)"
                    ) -> str:
        match = re.search(pattern, text_response, re.DOTALL)
        if match:
            answer_text = match.group(1).strip()
            return answer_text
        else:
            return text_response


class offine_rag:
    def __init__(self, llm) -> None:
        self.llm = llm
        self.promt = hub.pull("rlm/rag-prompt")
        seft.str_parser = Str_OutputParser()
    def get_chain(self, retriever):
        input_data = {
            "context": retriever | self.format_docs,
            "question": RunnablePassthrough()
        }
        rag_chain = (
        input_data
        | self.prompt
        | self.llm
        | self.str_parser
    )
        return rag_chain

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)