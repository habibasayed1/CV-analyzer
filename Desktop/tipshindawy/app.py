import os
import pdfplumber
import pytesseract
from pdf2image import convert_from_path

from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

app = Flask(__name__)
CORS(app)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=("AIzaSyDVQEFTSpYT8gLJ0JLC5GtT-OS4fk6s7Ho")
)

class CVAnalysis(BaseModel):
    skills: list[str]
    experience: list[str]

parser = JsonOutputParser(pydantic_object=CVAnalysis)

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        if text.strip():
            return text.strip()
    except Exception:
        pass

    try:
        images = convert_from_path(pdf_path)
        for image in images:
            text += pytesseract.image_to_string(image) + "\n"
    except Exception:
        pass

    return text.strip()

analysis_prompt = PromptTemplate(
    template="""
You are an HR assistant.

Extract from this CV:
- skills
- experience

Return ONLY valid JSON.

{format_instructions}

CV:
{cv}
""",
    input_variables=["cv"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

analysis_chain = analysis_prompt | llm | parser

idea_chain = PromptTemplate(
    template="Suggest suitable job roles based on:\n{analysis}",
    input_variables=["analysis"]
) | llm

job_chain = PromptTemplate(
    template="List top 5 job titles:\n{analysis}",
    input_variables=["analysis"]
) | llm

gap_chain = PromptTemplate(
    template="""
Compare CV with job description:

CV Analysis:
{analysis}

Job Description:
{job_desc}

Return table:
Skill | Exists | Needs Improvement
""",
    input_variables=["analysis", "job_desc"]
) | llm

def run_all_chains(cv_text):
    if not cv_text or cv_text.strip() == "":
        return {"error": "CV text is empty or unreadable"}

    try:
        analysis = analysis_chain.invoke({"cv": cv_text})

        return {
            "analysis": analysis,
            "jobs": job_chain.invoke({"analysis": analysis}).content,
            "ideas": idea_chain.invoke({"analysis": analysis}).content,
            "gap": gap_chain.invoke({
                "analysis": analysis,
                "job_desc": "Python developer with ML and SQL"
            }).content
        }

    except Exception as e:
        return {"error": str(e)}

@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("file")

    if not file:
        return jsonify({"error": "No file uploaded"})

    path = "temp.pdf"
    file.save(path)

    text = extract_text_from_pdf(path)
    result = run_all_chains(text)

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)