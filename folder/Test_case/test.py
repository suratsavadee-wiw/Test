#code one
import whisper
import os
import re
import csv
import pandas as pd
import jiwer
import torch
from transformers import pipeline
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv
# from pythainlp.spell import correct
# from pythainlp.tokenize import word_tokenize

# import pythainlp
# print(pythainlp.__version__)

load_dotenv()
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

# Step 1: use Whisper for speech-to-text
MODEL_NAME = "biodatlab/whisper-th-medium-combined"
device = 0 if torch.cuda.is_available() else "cpu"
pipe = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_NAME,
    chunk_length_s=30,
    device=device,
)

audio_file = "/Users/momo/Downloads/Telesales.wav"
transcription = pipe(
    audio_file,
    generate_kwargs={"language": "<|th|>", "return_timestamps": False}, 
)["text"]

# # Save cleaned transcription
# with open("transcription_cleaned.txt", "w", encoding="utf-8") as txt_file:
#     txt_file.write(transcription)

#Save as txt
with open("transcription.txt", "w", encoding="utf-8") as txt_file:
    txt_file.write(transcription)

#Save as csv
csv_filename = "transcription.csv"
with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Transcription"])
    writer.writerow([transcription])

#Step 2: Evaluate Speech Recognition (WER & CER)
def evaluate_transcription(ground_truth, predicted_text):
    wer = jiwer.wer(ground_truth, predicted_text)
    cer = jiwer.cer(ground_truth, predicted_text)
    return {"WER": wer, "CER": cer}

#Evaluation 
ground_truth = "ขอให้คุณลูกค้ามีสุขภาพแข็งแรงตลอดปี"
eval_result = evaluate_transcription(ground_truth, transcription)
print("Speech Recognition Evaluation:", eval_result)

#Step 3: Chunking
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n"],  
    chunk_size=1000,    
    chunk_overlap=200,  
    length_function=len,
)
chunks = text_splitter.split_text(transcription)
documents = [Document(page_content=chunk) for chunk in chunks]

#Step 4: Vectorization
embedding_azure_deployment = "text-embedding-3-large"
openai_api_version = "2024-02-01"
llm_azure_deployment = "gpt-4o"
model_version = "2024-05-13"

embeddings = AzureOpenAIEmbeddings(
    openai_api_version=openai_api_version,
    azure_deployment=embedding_azure_deployment,
)
vectorstore = Chroma.from_documents(documents, embeddings, persist_directory="./chroma_db")
# print("Vectorization complete")

#Step 5: LLM for Sales Coaching & Feature Extraction
llm = AzureChatOpenAI(
    openai_api_version=openai_api_version,
    azure_deployment=llm_azure_deployment,
    model_version=model_version,
    temperature=0,
)

retrieval_prompt = """
You are an AI assistant analyzing a sales call.
Use the retrieved documents to answer concisely.
<Query>
{question}
</Query>
<Context>
{context}
</Context>
<Answer>
"""

prompt_template = ChatPromptTemplate.from_template(retrieval_prompt)

def format_docs(docs):
    return "\n".join([doc.page_content for doc in docs])

rag_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    | prompt_template
    | llm
    | StrOutputParser()
)

rag_chain_with_source = RunnableParallel(
    {
        "context": lambda query: vectorstore.similarity_search(query, k=5),
        "question": RunnablePassthrough()
    }
).assign(answer=rag_chain_from_docs)

#Step 6: Feature Extraction Evaluation
def evaluate_feature_extraction(transcribed_text, extracted_features):
    relevant_keywords = ["เงินคืน", "รับประกัน", "การันตี"]
    extracted_count = sum(1 for word in relevant_keywords if word in extracted_features.lower())
    return {"Relevant Features Found": extracted_count, "Total Features Expected": len(relevant_keywords)}

#Extraction Evaluation
extracted_features = "เงินคืนการันตีทุกปี 24 ปี"
feature_eval = evaluate_feature_extraction(transcription, extracted_features)
print("Feature Extraction Evaluation:", feature_eval)

#step 7: Sales Coaching Evaluation
def evaluate_sales_coaching(question, expected_response):
    response = llm.invoke(question)
    similarity = jiwer.wer(expected_response, response)
    return {"WER Similarity Score": similarity, "LLM Response": response}

#Sales Coaching Evaluation
question = "จะเชิญชวนให้ลูกค้าซื้อกรมธรรมได้อย่างไร?"
expected_response = "จะเชิญชวนโดยการเน้นถึงผลประโยชน์และความคุ้มครอง"
sales_eval = evaluate_sales_coaching(question, expected_response)
print("Sales Coaching Evaluation:", sales_eval)



