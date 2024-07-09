from django.shortcuts import render
from django.shortcuts import render, redirect

from .models import Transcripts
from django.shortcuts import render, redirect
from .models import Transcripts
from Bio_Epidemiology_NER.bio_recognizer import ner_prediction

medical_text = """
Upon examination, the patient presented with symptoms of persistent coughing, chest congestion, and occasional fever. Further investigation revealed signs of respiratory distress and decreased lung function, indicative of a possible respiratory infection. After careful consideration, I have diagnosed the patient with bronchitis. Treatment will involve a course of antibiotics to target the underlying bacterial infection, along with bronchodilators to alleviate airway constriction and corticosteroids to reduce inflammation. Additionally, I recommend plenty of rest and hydration to support the body's healing process. It's crucial for the patient to adhere to the prescribed medication regimen and follow up for monitoring to ensure a speedy recovery. Also, I've scheduled a follow-up appointment for the patient on 17th March at 11.30. During this session, we'll assess the response to the prescribed medication regimen, monitor any changes in symptoms, and discuss further steps in managing the condition. It's essential for the patient to attend this appointment to ensure proper monitoring and adjustment of treatment as needed. Furthermore, I advise the patient to avoid exposure to smoke and other respiratory irritants to prevent exacerbation of symptoms
"""

from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from transformers import pipeline

# Define your Groq API key
GROQ_API_KEY = "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"
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model="Llama3-8b-8192")

# Function to generate responses
def answer_query(question, context):
    # Create text chunks and generate embeddings
    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=2000, chunk_overlap=200)
    documents = text_splitter.create_documents([context])

    embeddings = HuggingFaceEmbeddings()

    # Create a FAISS index
    faiss_index = FAISS.from_documents(documents, embeddings)

    # Define the prompt template with simplified explanations
    prompt_template = """
    You are a helpful assistant who explains complex medical terms in extremely simplified language suitable for someone with no medical background. Below is the medical information for a patient. For each term, provide a clear and simple explanation.

    If the question asks for a specific detail, such as the patient's name or age, provide only that detail without additional information.

    Examples:
    1. Medical text:
       Name: Mr. Johnson
       Symptoms: Fatigue, dizziness, shortness of breath
       Diagnosis: Faulty heart valve

       Question: What is a heart valve?
       Helpful Answer: A heart valve is a door-like structure in the heart that controls the flow of blood. If it becomes faulty, it may cause problems with blood flow.

    2. Medical text:
       Name: Mr. Johnson
       Symptoms: Fatigue, dizziness, shortness of breath
       Diagnosis: Faulty heart valve

       Question: What is valve replacement surgery?
       Helpful Answer: Valve replacement surgery is a procedure to replace a damaged or faulty heart valve with an artificial or biological valve.

    Please provide similar explanations for the following terms based on the context:
    {context}

    Question: {question}
    Helpful Answer:
    """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Set up RAG with the Model
    knowledge_base = faiss_index

    # Set up RetrievalQA with the correct prompt template
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=knowledge_base.as_retriever(),
        chain_type_kwargs={"prompt": PROMPT}
    )

    # Get response from the model
    response = qa_chain({"context": context, "question": question, "query":question})
    return response["result"]


def doctor(request):
    if request.method == "POST":
        text = request.POST.get('text')
        transript_section = Transcripts(text=text)
        transript_section.save()
        return redirect('main:patient')

    return render(request, 'doctor.html')

def patient(request):
    question5 = "Provide a complete summary of each item in the section diagnosis and also items from similarly meaning sections"
    response5 = answer_query(question5, medical_text)
    print("Response 5 :", response5)
    return render(request, 'patient.html', {'response': response5})
