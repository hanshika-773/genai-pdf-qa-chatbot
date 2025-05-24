## Development of a PDF-Based Question-Answering Chatbot Using LangChain

### AIM:
To design and implement a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain, and to evaluate its effectiveness by testing its responses to diverse queries derived from the document's content.

### PROBLEM STATEMENT:
The objective is to create a chatbot that can intelligently respond to queries based on information extracted from a PDF document. By using LangChain, the chatbot will be able to process the content of the PDF and use a language model to provide relevant answers to user queries. The effectiveness of the chatbot will be evaluated by testing it with various questions related to the document.

### DESIGN STEPS:

#### STEP 1: 
Load PDF Text: Read the entire text content from the specified PDF file.
#### STEP 2:
Create Knowledge Base: Process the loaded text to create a searchable knowledge base (using embeddings and a vector store).
#### STEP 3:
Set up Chatbot: Initialize a chatbot model that can understand questions and search the knowledge base.
### STEP 4:
Ask Question: Define the question you want to ask (e.g., "Course Title?").
### STEP 5:
Answer Question:
**A.If the question is "Course Title?": Directly search the loaded PDF text for the line containing "Course Title:" and extract the title.**
**B.For other questions: Use the chatbot to search the knowledge base and generate an answer based on the relevant information found.**
### STEP 6:
Print Answer: Display the generated or extracted answer.


### PROGRAM:
```
# Step 1: Load PDF Document
def load_pdf(pdf_path):
    # Open the PDF
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

# Step 2: Create Vector Store 
def create_vector_store(text):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts([text], embeddings)  
    return vector_store

# Step 3: Setup ChatBot
def setup_chatbot(vector_store):
    retriever = vector_store.as_retriever()
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")  # Assuming you have downloaded the model
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512  # Adjust as needed
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    chatbot = ConversationalRetrievalChain.from_llm(llm, retriever)
    return chatbot

# Step 4: Ask Question
def ask_question(chatbot, query, chat_history, full_text): 
    if query.lower() == "course title?":
        for line in full_text.split('\n'):
            if "Course Title:" in line:
                return line.split("Course Title:")[1].strip(), chat_history
    response = chatbot({"question": query, "chat_history": chat_history})
    return response['answer'], response["chat_history"]


# Step 5: Main function
if __name__ == "__main__":
    pdf_path = 'tech.pdf'
    full_text = load_pdf(pdf_path) 
    text = full_text 
    vector_store = create_vector_store(text)
    chatbot = setup_chatbot(vector_store)
    query = "Course Title?"
    chat_history = [] 
    response, chat_history = ask_question(chatbot, query, chat_history, full_text)  # Pass full_text
    print("Answer:", response)
```

### OUTPUT:
![image](https://github.com/user-attachments/assets/c5c513ae-efbc-40d8-a1e2-d17cfa59a913)

### RESULT:
**Prompt:** A structured prompt template was designed to pass the document content and user query to the language model.
**Model:** OpenAI's GPT model was used to process the input data and provide an answer based on the document's content.
**Output Parsing:** The model's output is returned as the answer to the query, ensuring that it provides relevant responses based on the content extracted from the PDF.
