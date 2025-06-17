from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

embedding = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = FAISS.from_documents(documents, embedding)
vectorstore.save_local("faiss_index_hr")