import json
import sys
import time

from dotenv import load_dotenv
import requests
import pdfplumber

from langchain import callbacks
from langchain.schema import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_chroma import Chroma

import numpy as np
from sentence_transformers import SentenceTransformer

# ==============================================================================
# !!! 警告 !!!: 以下の変数を変更しないでください。
# ==============================================================================
model = "gpt-4o-mini"

# 本来はGCSのURLでしたが、一応置換
pdf_file_urls = [
    "pdf_1.pdf",
    "pdf_2.pdf",
    "pdf_3.pdf",
    "pdf_4.pdf",
    "pdf_5.pdf",
]
# ==============================================================================


# ==============================================================================
# この関数を編集して、あなたの RAG パイプラインを実装してください。
# !!! 注意 !!!: デバッグ過程は標準出力に出力しないでください。
# ==============================================================================
def rag_implementation(question: str) -> str:
    """
    ロート製薬の製品・企業情報に関する質問に対して回答を生成する関数
    この関数は与えられた質問に対してRAGパイプラインを用いて回答を生成します。

    Args:
        question (str): ロート製薬の製品・企業情報に関する質問文字列

    Returns:
        answer (str): 質問に対する回答

    Note:
        - デバッグ出力は標準出力に出力しないでください
        - model 変数 と pdf_file_urls 変数は編集しないでください
        - 回答は日本語で生成してください
    """
    def download_and_load_pdfs(urls: list) -> list:
        """
        PDFファイルをダウンロードして読み込む関数

        Args:
            urls (list): PDFファイルのURLリスト

        Returns:
            documents (list): PDFファイルのテキストデータを含むDocumentオブジェクトのリスト

        Raises:
            Exception: ダウンロードまたは読み込みに失敗した場合に発生する例外

        Examples:
            >>> urls = ["https://example.com/example.pdf"]
            >>> download_and_load_pdfs(urls)
            [Document(page_content="...", metadata={"source": "https://example.com/example.pdf"})]
        """
        try:
            def download_pdf(url, save_path):
                response = requests.get(url)
                if response.status_code == 200:
                    with open(save_path, 'wb') as f:
                        f.write(response.content)
                else:
                    raise Exception(f"Failed to download {url}")
            documents = []

            for i, url in enumerate(urls):
                tmp_path = f"pdf_{i}.pdf"
                download_pdf(url, tmp_path)

                with pdfplumber.open(tmp_path) as pdf:
                    full_text = ""
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            full_text += text + "\n"

                    documents.append(
                        Document(
                            page_content=full_text,
                            metadata={"source": url}
                        )
                    )
            return documents
        except Exception as e:
            raise Exception(f"Error reading {url}: {e}")

    def create_vectorstore(docs: list) -> Chroma:
        """
        テキストデータからベクトルストアを生成する関数

        Args:
            docs (list): Documentオブジェクトのリスト

        Returns:
            vectorstore (Chroma): ベクトルストア

        Raises:
            Exception: ベクトルストアの生成に失敗した場合に発生する例外

        Examples:
            >>> docs = [Document(page_content="...", metadata={"source": "https://example.com/example.pdf"})]
            >>> create_vectorstore(docs)
            Chroma(...)
        """
        try:
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
            )
            splitted_docs = []
            for doc in docs:
                chunks = text_splitter.split_text(doc.page_content)
                for chunk in chunks:
                    splitted_docs.append(Document(page_content=chunk, metadata=doc.metadata))

            embedding_function = OpenAIEmbeddings()

            vectorstore = Chroma.from_documents(
                splitted_docs,
                embedding_function,
            )
            return vectorstore
        except Exception as e:
            raise Exception(f"Error creating vectorstore: {e}")

    class SentenceTransformerRetriever:
        def __init__(self, documents, top_k=4):
            self.documents = documents
            self.top_k = top_k
            self.model = SentenceTransformer("hotchpotch/static-embedding-japanese", device="cpu")

            # 文書を分割してエンコード
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
            )
            self.chunks = []
            for doc in documents:
                splits = text_splitter.split_text(doc.page_content)
                for split in splits:
                    self.chunks.append(Document(page_content=split, metadata=doc.metadata))

            # 文書をエンコード
            self.embeddings = self.model.encode([chunk.page_content for chunk in self.chunks])

        def get_relevant_documents(self, query):
            # クエリをエンコード
            query_embedding = self.model.encode(query)

            # コサイン類似度を計算
            similarities = np.dot(self.embeddings, query_embedding) / (
                np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
            )

            # 上位k個の類似度を持つドキュメントを取得
            top_indices = np.argsort(similarities)[-self.top_k:][::-1]

            return [self.chunks[i] for i in top_indices]

    # main関数
    docs = download_and_load_pdfs(pdf_file_urls)
    retriever = SentenceTransformerRetriever(docs)

    template = """
# ゴール
私は、参考文章と質問を提供します。
あなたは、参考文章に基づいて、質問に対する解答を生成してください。
ステップ・バイ・ステップで考えてください。また、解答は端的な解答でお願いします。
# 質問
{question}
# 参考文章
{context}
"""
    prompt = ChatPromptTemplate.from_template(template)
    chat = ChatOpenAI(model=model)
    output_parser = StrOutputParser()

    setup_and_retrieval = RunnableParallel(
        {"context": lambda x: "\n".join([doc.page_content for doc in retriever.get_relevant_documents(x)]),
         "question": RunnablePassthrough()}
    )

    chain = setup_and_retrieval | prompt | chat | output_parser
    answer = chain.invoke(question)
    return answer


# ==============================================================================


# ==============================================================================
# !!! 警告 !!!: 以下の関数を編集しないでください。
# ==============================================================================
def main(question: str):
    with callbacks.collect_runs() as cb:
        result = rag_implementation(question)
        for attempt in range(2):  # 最大2回試行
            try:
                run_id = cb.traced_runs[0].id
                break
            except IndexError:
                if attempt == 0:  # 1回目の失敗時のみ
                    time.sleep(3)  # 3秒待機して再試行
                else:  # 2回目も失敗した場合
                    raise RuntimeError("Failed to get run_id after 2 attempts")

    output = {"result": result, "run_id": str(run_id)}
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    load_dotenv()

    if len(sys.argv) > 1:
        question = sys.argv[1]
        main(question)
    else:
        print("Please provide a question as a command-line argument.")
        sys.exit(1)
# ==============================================================================
