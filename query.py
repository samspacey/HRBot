from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

llm = ChatOpenAI(model="gpt-4o", temperature=0)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type="stuff"
)

query = "How many days of sick leave am I entitled to?"
response = qa(query)

print(response["result"])
for doc in response["source_documents"]:
    print(f"\nSource:\n{doc.page_content[:300]}")