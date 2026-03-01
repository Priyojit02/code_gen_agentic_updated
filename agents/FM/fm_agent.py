import os
import re
import json
import uuid
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from agents.base_agent import BaseAgent

load_dotenv()
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

class FmAgent(BaseAgent):
    """
    Generates SAP ABAP Function Module definitions dynamically using LLM + RAG context.
    """

    # ------------------ LLM INIT ------------------
    def _init_llm(self):
        """Initialize the LLM client."""
        return ChatOpenAI(
            model_name=os.getenv("FM_MODEL_NAME", "gpt-5"),
            temperature=float(os.getenv("FM_TEMPERATURE", "0.1")),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

    # ------------------ VECTORSTORE INIT ------------------
    def _init_vectorstore(self):
        """
        Loads or builds FAISS vector DB from fm_requirements.txt file.
        """
        kb_path = Path(os.path.dirname(__file__)) / "FM_RAG_KB.txt"
        vs_path = Path(os.path.dirname(__file__)) / "fm_vector_store"

        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

        if vs_path.exists():
            kb_mtime = kb_path.stat().st_mtime if kb_path.exists() else 0
            vs_mtime = max((f.stat().st_mtime for f in vs_path.glob("**/*")), default=0)
            if kb_mtime <= vs_mtime:
                self.logger.info("📚 Loading existing FAISS vector DB for FmAgent...")
                return FAISS.load_local(vs_path, embeddings, allow_dangerous_deserialization=True)
            else:
                self.logger.info("🔄 KB updated — rebuilding FAISS index...")

        if not kb_path.exists():
            self.logger.warning(f"⚠️ No KB file found at {kb_path}. Proceeding without RAG context.")
            return None

        kb_text = kb_path.read_text(encoding="utf-8").strip()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = [Document(page_content=chunk) for chunk in splitter.split_text(kb_text)]

        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(vs_path)
        self.logger.info("✅ New FAISS vector DB created for FmAgent.")
        return vectorstore

    def _get_relevant_context(self, query: str, k: int = 4) -> str:
        """Retrieves top-k relevant context chunks from FAISS vector store."""
        if not hasattr(self, "vectorstore") or self.vectorstore is None:
            self.vectorstore = self._init_vectorstore()
        if not self.vectorstore:
            return ""

        results = self.vectorstore.similarity_search(query, k=k)
        if not results:
            return ""

        combined = "\n\n".join(r.page_content for r in results)
        self.logger.info(f"📖 Retrieved {len(results)} RAG context chunks.")
        return combined

    # ------------------ MAIN RUN METHOD ------------------
    def run(self, section_text: str, purposes: dict | None = None, metadata=None) -> dict:
        """
        Main function to generate ABAP Function Module.
        """
        if not section_text:
            self.logger.warning("No text provided to FmAgent.")
            return {"type": "fm", "purpose": "", "code": ""}

        self.logger.info("Running FmAgent with provided section text...")
        print("\n🧠 [FmAgent] Received text:\n", section_text[:500], "\n--- END ---")

        # --- Retrieve RAG context ---
        rag_context = self._get_relevant_context(section_text)
        full_context = section_text.strip()

        # Merge purposes (like class, cds, etc.)
        if purposes:
            self.logger.info(f"🔍 Received purposes from previous agents: {list(purposes.keys())}")
            for key, text in purposes.items():
                if text and len(text.strip()) > 25:
                    full_context += f"\n\nAdditional Context ({key}):\n{text.strip()}"
                    
        if rag_context:
            full_context += f"\n\n--- Retrieved Knowledge Base Context ---\n{rag_context}"

        # --- Extract metadata ---
        import_params = metadata.get("import_params") if metadata else None
        export_params = metadata.get("export_params") if metadata else None

        # --- System message ---
        system_message = SystemMessage(
            content=(
                "You are a senior SAP ABAP developer. "
                "Generate Function Module code following SAP naming conventions, "
                "best practices, and clear modularization. "
                "Use IMPORTING and EXPORTING parameters properly, "
                "and include example SELECT statements if needed."
            )
        )

        # --- Prompt to LLM ---
        prompt = f"""
        Based on the following requirement, generate JSON with two keys:
        1) "fm_code": Complete SAP ABAP Function Module source code.
        2) "fm_purpose": Short explanation of its purpose.

        Requirement Context:
        {full_context}
        
        --- Retrieved RAG Knowledge Base ---
        {rag_context if rag_context else "No additional RAG context found."}

        Import Parameters:
        {json.dumps(import_params, indent=2) if import_params else "None"}

        Export Parameters:
        {json.dumps(export_params, indent=2) if export_params else "None"}

        Rules:
         - Ensure clean and compilable ABAP syntax.
         - Define IMPORTING and EXPORTING parameters in the interface.
         - Implement ABAP logic that uses import parameters to fetch or process data.
         - Return processed or selected data into export parameters.
         - Create FM as per the requirements

        Output JSON only, strictly formatted:
        {{
            "fm_code": "...",
            "fm_purpose": "..."
        }}
        """

        # --- Invoke LLM ---
        resp = self.llm.invoke([system_message, HumanMessage(content=prompt)])
        raw = getattr(resp, "content", str(resp)).strip()

        # --- Parse JSON ---
        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                raise ValueError("No JSON object found in LLM response.")
            data = json.loads(match.group())
        except Exception as e:
            self.logger.error(f"Failed to parse JSON from LLM output: {e}")
            self.logger.debug(f"Raw LLM output:\n{raw}")
            raise

        fm_code = data.get("fm_code", "").strip()
        fm_purpose = data.get("fm_purpose", "").strip()

        # --- Save result ---
        # fm_file = self.job_dir / f"FunctionModule_{uuid.uuid4().hex}.abap"
        # fm_file.write_text(fm_code, encoding="utf-8")

        # self.logger.info(f"✅ Function Module saved to: {fm_file}")
        self.logger.debug(f"Extracted FM Purpose: {fm_purpose[:200]}...")

        return {
            "type": "fm",
            "purpose": fm_purpose,
            "code": fm_code,
        }