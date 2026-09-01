import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# Replace GoogleGenerativeAIEmbeddings with this:
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set in .env file.")

os.makedirs("data", exist_ok=True)

kb_content = """MYFINERGY COMPANY KNOWLEDGE BASE

--- OVERVIEW & OBJECTIVES ---
Website: https://www.myfinergy.com/
Business Goal: MyFinergy is a SaaS diagnostic product for financial advisors that solves client prospecting and onboarding problems. It builds client trust in 5 minutes and captures client financial goals so advisors can offer advice first rather than just pushing financial products.
Pricing: ₹1,899 per month with a minimum commitment of 1 year.

--- TARGET AUDIENCE ---
MyFinergy is built for licensed financial professionals holding IRDA licenses, AMFI/NISM Mutual Fund certifications, or RIA credentials who distribute Life Insurance, Health Insurance, Mutual Funds, and Investment Products.

--- THE T4 ONBOARDING FRAMEWORK ---
MyFinergy uses the T4 Framework to onboard clients:
- T1 (TEACH): Educational tools (inflation calculators, budget tools, risk graphs) to educate prospects on financial goals.
- T2 (TEST): Diagnostic tools to test client readiness and trigger self-realization of insurance and investment needs.
- T3 (TREATMENT): Generates comprehensive financial fitness reports and a 12-page diagnostic output to recommend suitable financial products.
- T4 (TRACK): Goal tracking tools to monitor client progress over time.

--- PAIN POINTS SOLVED FOR ADVISORS ---
1. Qualified Lead Generation: Captures actual financial goals along with contact details because client trust is built first.
2. Conversion: Converts prospects by presenting diagnostic data and risk graphs rather than aggressive sales tactics.
3. Referrals: Encourages existing clients to refer their advisor to friends and family.
4. Personal Branding: Transforms normal commission agents into respected, professional advisors.

--- COMPETITORS & POSITIONING ---
Competitors: InvestWell, REDVision, Fintso, IFA Central, MProfit, GoalTeller, AssetPlus, Wealth Elite, NJ Wealth, Prudent, Invest4Edu.
Positioning Difference: Non-subscribed advisors often mistake operational software for sales platforms. Competitors are operational practice-management tools or single-category software. MyFinergy is a comprehensive financial diagnostic platform covering all product lines (Insurance + Mutual Funds + Wealth).

--- COMPLIANCE & LEGAL GUARDRAILS ---
- Diagnostic Tool Only: MyFinergy is a financial diagnostic tool that provides raw data and reports; it is NOT a financial planning or execution platform.
- No Stock or Fund Advice: MyFinergy does not recommend specific stocks, mutual funds, or insurance policies to retail clients.
"""

kb_path = "data/knowledge_base.txt"
with open(kb_path, "w", encoding="utf-8") as f:
    f.write(kb_content)

print("SUCCESS: data/knowledge_base.txt created!")

loader = TextLoader(kb_path, encoding="utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Fix: Standard model name with explicit API key reference
embeddings = GoogleGenerativeAIEmbeddings(
    model="text-embedding-004",
    google_api_key=api_key
)

persist_directory = "./chroma_db"
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=persist_directory
)

print("SUCCESS: Knowledge base embedded into ./chroma_db!")