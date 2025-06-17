from os import listdir
from os.path import join
from langchain_community.document_loaders import PyPDFLoader

for fn in listdir("./policies"):
    if not fn.lower().endswith(".pdf"):
        continue
    path = join("./policies", fn)
    for doc in PyPDFLoader(path).load():
        if "sick" in doc.page_content.lower():
            print(f"{fn} (page {doc.metadata['page']}):")
            print(doc.page_content[:200].replace("\n"," "))
            print("---")
