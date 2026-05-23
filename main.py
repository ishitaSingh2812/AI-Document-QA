from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

app = FastAPI()

db = None


@app.get("/", response_class=HTMLResponse)
def home():

    return """

    <html>

    <head>

        <title>AI Document QA</title>

        <style>

            body{
                font-family: Arial;
                background: #f5f5f5;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
            }

            .container{
                width: 500px;
                background: white;
                padding: 30px;
                border-radius: 12px;
                box-shadow: 0px 0px 15px rgba(0,0,0,0.2);
                text-align: center;
            }

            h1{
                margin-bottom: 20px;
            }

            input{
                width: 100%;
                padding: 12px;
                margin-top: 15px;
                border-radius: 6px;
                border: 1px solid #ccc;
            }

            button{
                width: 100%;
                padding: 12px;
                margin-top: 15px;
                border: none;
                background: #007bff;
                color: white;
                border-radius: 6px;
                cursor: pointer;
                font-size: 16px;
            }

            button:hover{
                background: #0056b3;
            }

            #answer{
                margin-top: 20px;
                background: #eeeeee;
                padding: 15px;
                border-radius: 6px;
                text-align: left;
            }

        </style>

    </head>

    <body>

        <div class="container">

            <h1>AI Document QA</h1>

            <input type="file" id="pdfFile">

            <button onclick="uploadPDF()">Upload PDF</button>

            <input type="text" id="question" placeholder="Ask a question">

            <button onclick="askQuestion()">Get Answer</button>

            <div id="answer"></div>

        </div>

        <script>

            async function uploadPDF(){

                const fileInput = document.getElementById("pdfFile");

                if(fileInput.files.length === 0){
                    alert("Please select a PDF");
                    return;
                }

                const formData = new FormData();

                formData.append("file", fileInput.files[0]);

                const response = await fetch("/upload", {
                    method: "POST",
                    body: formData
                });

                const data = await response.json();

                alert(data.message);
            }

            async function askQuestion(){

                const question = document.getElementById("question").value;

                if(question === ""){
                    alert("Please enter a question");
                    return;
                }

                const response = await fetch(`/ask?question=${question}`);

                const data = await response.json();

                document.getElementById("answer").innerHTML = data.answer;
            }

        </script>

    </body>

    </html>

    """


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    global db

    try:

        # Save uploaded PDF
        with open("uploaded.pdf", "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Load PDF
        loader = PyPDFLoader("uploaded.pdf")
        docs = loader.load()

        # Split document
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = splitter.split_documents(docs)

        # Create embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Store in ChromaDB
        db = Chroma.from_documents(chunks, embeddings)

        return {"message": "PDF uploaded successfully ✅"}

    except Exception as e:

        return {"message": str(e)}


@app.get("/ask")
def ask(question: str):

    global db

    if db is None:
        return {"answer": "Please upload a PDF first"}

    retriever = db.as_retriever(search_kwargs={"k": 3})

    docs = retriever.get_relevant_documents(question)

    answer = ""

    for doc in docs:
        answer += doc.page_content + "\n"

    return {"answer": answer}
