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
    # [Previous download_and_load_pdfs function remains the same]
    def download_and_load_pdfs(urls: list) -> list:
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
        try:
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=250,
            )
            splitted_docs = []
            for doc in docs:
                chunks = text_splitter.split_text(doc.page_content)
                for chunk in chunks:
                    splitted_docs.append(Document(page_content=chunk, metadata=doc.metadata))
            embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
            vectorstore = Chroma.from_documents(
                splitted_docs,
                embedding_function,
            )
            return vectorstore
        except Exception as e:
            raise Exception(f"Error creating vectorstore: {e}")

    class SentenceTransformerRetriever:
        def __init__(self, documents, top_k=5):
            self.documents = documents
            self.top_k = top_k
            self.model = SentenceTransformer("hotchpotch/static-embedding-japanese", device="cpu")

            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=250,
            )
            self.chunks = []
            for doc in documents:
                splits = text_splitter.split_text(doc.page_content)
                for split in splits:
                    self.chunks.append(Document(page_content=split, metadata=doc.metadata))

            self.embeddings = self.model.encode([chunk.page_content for chunk in self.chunks])

        def get_relevant_documents(self, query):
            query_embedding = self.model.encode(query)

            similarities = np.dot(self.embeddings, query_embedding) / (
                np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
            )

            top_indices = np.argsort(similarities)[-self.top_k:][::-1]

            return [self.chunks[i] for i in top_indices]

    def combine_retrieval_results(st_docs, vs_docs, sentence_transformer_model, num_results=4):
        def calculate_cosine_similarity(doc1, doc2, model):
            # 文書のエンベディングを計算
            emb1 = model.encode(doc1.page_content)
            emb2 = model.encode(doc2.page_content)

            # コサイン類似度を計算
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            return similarity

        combined_docs = []
        # st_docsとvs_docsを交互に見ていく
        st_idx = 0
        vs_idx = 0

        while len(combined_docs) < num_results and (st_idx < len(st_docs) or vs_idx < len(vs_docs)):
            current_doc = None
            # SentenceTransformerの結果を確認
            if st_idx < len(st_docs):
                st_doc = st_docs[st_idx]
                is_duplicate = False

                # 既存の文書との類似度チェック
                for used_doc in combined_docs:
                    similarity = calculate_cosine_similarity(st_doc, used_doc, sentence_transformer_model)
                    if similarity > 0.9:  # コサイン類似度のしきい値
                        is_duplicate = True
                        break

                if not is_duplicate:
                    current_doc = st_doc
                st_idx += 1

            # 現在のターンでまだ文書が選択されていない場合、VectorStoreの結果を確認
            if current_doc is None and vs_idx < len(vs_docs):
                vs_doc = vs_docs[vs_idx]
                is_duplicate = False

                # 既存の文書との類似度チェック
                for used_doc in combined_docs:
                    similarity = calculate_cosine_similarity(vs_doc, used_doc, sentence_transformer_model)
                    if similarity > 0.9:  # コサイン類似度のしきい値
                        is_duplicate = True
                        break

                if not is_duplicate:
                    current_doc = vs_doc
                vs_idx += 1

            # 重複していない文書が見つかった場合、追加
            if current_doc is not None:
                combined_docs.append(current_doc)

        return combined_docs

    # main処理
    docs = download_and_load_pdfs(pdf_file_urls)

    # 両方のretrieverを初期化
    st_retriever = SentenceTransformerRetriever(docs)
    vectorstore = create_vectorstore(docs)

    # 両方の手法で文書を取得
    st_docs = st_retriever.get_relevant_documents(question)
    vs_docs = vectorstore.similarity_search(question, k=5)

    # 結果を組み合わせ
    combined_docs = combine_retrieval_results(st_docs, vs_docs, st_retriever.model, num_results=4)

    template = """
# ゴール
私は、参考文章と質問を提供します。
あなたは、参考文章に基づいて、質問に対する解答を生成してください。
ステップ・バイ・ステップで考えてください。また、質問への解答を端的に出力してください。
# 質問
{question}
# 参考文章
{context}
"""
    prompt = ChatPromptTemplate.from_template(template)
    chat = ChatOpenAI(model=model, temperature=0.1)
    output_parser = StrOutputParser()

    setup_and_retrieval = RunnableParallel(
        {"context": lambda x: "\n".join([doc.page_content for doc in combined_docs]),
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
