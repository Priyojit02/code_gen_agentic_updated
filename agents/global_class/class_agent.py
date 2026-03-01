import os
import re, asyncio
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


class ClassAgent(BaseAgent):
    """
    Generates and refines SAP ABAP GLOBAL classes using RAG-based guidance.
    Also provides a detailed 'purpose' description of the class and its methods.
    """

    def _init_llm(self):
        return ChatOpenAI(
            model_name=os.getenv("CLASS_MODEL_NAME", "gpt-5"),
            temperature=float(os.getenv("CLASS_TEMPERATURE", "0.1")),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

    # --- Vector DB Setup ---
    def _init_vectorstore(self):
        """
        Loads or builds FAISS vector DB from the Class_RAG_KB.txt file.
        """
        kb_path = Path(os.path.dirname(__file__)) / "Class_RAG_KB.txt"
        vs_path = Path(os.path.dirname(__file__)) / "rag_vector_store_class"
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

        # Reuse if KB not updated
        if vs_path.exists():
            kb_mtime = kb_path.stat().st_mtime if kb_path.exists() else 0
            vs_mtime = max(f.stat().st_mtime for f in vs_path.glob("**/*"))
            if kb_mtime <= vs_mtime:
                self.logger.info("📚 Loading existing FAISS vector DB for ClassAgent...")
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
        self.logger.info("✅ New FAISS vector DB created for ClassAgent.")
        return vectorstore

    def _get_relevant_context(self, query: str, k: int = 4) -> str:
        """
        Retrieves top-k relevant KB chunks from vector DB for given query.
        """
        if not hasattr(self, "vectorstore") or self.vectorstore is None:
            self.vectorstore = self._init_vectorstore()
        if not self.vectorstore:
            return ""

        results = self.vectorstore.similarity_search(query, k=k)
        if not results:
            return ""
        combined = "\n\n".join([r.page_content for r in results])
        self.logger.info(f"📖 Retrieved {len(results)} RAG context chunks for class.")
        return combined
    # -------------------------------------------------------------------------
    # 🧩 Retry logic with 600s timeout for LLM calls
    # -------------------------------------------------------------------------
    async def _call_llm_with_retry(self, messages, max_retries=3, timeout=900):
        """
        Calls the LLM up to max_retries times with timeout for each attempt.
        """
        for attempt in range(1, max_retries + 1):
            try:
                self.logger.info(f"🌀 Attempt {attempt}/{max_retries} - Calling LLM...")
                result = await asyncio.wait_for(self.llm.agenerate([messages]), timeout=timeout)
                text = result.generations[0][0].text.strip()
                return text
            except asyncio.TimeoutError:
                self.logger.warning(f"⏰ Timeout after {timeout}s on attempt {attempt}")
            except Exception as e:
                self.logger.error(f"⚠️ LLM call failed on attempt {attempt}: {e}")
            await asyncio.sleep(2)
        return "[Error: All LLM attempts failed after 3 retries.]"
    # --- Main Execution ---
    def run(self, section_text: str, purposes: dict | None = None, metadata=None) -> dict:
        if not section_text:
            self.logger.warning("No text provided to ClassAgent.")
            return {"purpose": "", "code": ""}
        # print(section_text)
        full_context = section_text.strip()

        # Merge purposes from previous agents
        if purposes:
            self.logger.info(f"🔍 Received purposes from previous agents: {list(purposes.keys())}")
            seen_texts = set()
            for key, text in purposes.items():
                if not text or not text.strip():
                    continue
                clean = text.strip()
                if clean in seen_texts:
                    continue
                seen_texts.add(clean)
                full_context += f"\n\nAdditional Context ({key}):\n{clean}"

        # --- Retrieve RAG context ---
        rag_context = self._get_relevant_context(full_context)
        if rag_context:
            full_context += f"\n\n--- Retrieved Knowledge Base Context ---\n{rag_context}"

        self.logger.info("🏗️ Running ClassAgent for global class generation...")

        # --- System role ---
        system_message = SystemMessage(
            content=(
                "You are an expert SAP ABAP OOP developer. "
                "Generate well-structured, production-grade GLOBAL ABAP classes. "
                "Use naming convention ZCL_<FUNCTION> for the class name. "
                "Follow SAP best practices: PUBLIC, PROTECTED, PRIVATE sections, method definitions, and exception handling. "
                "Ensure code is ready to be activated directly in Eclipse (no markdown, no explanations). "
                "Do not use local classes or includes — only global class definition and implementation."
            )
        )

        # --- Step 1: Generate Class Code ---
        draft_prompt = f"""
        Generate a global ABAP class (ZCL_*) based on the following requirement.
        Include:
        - Class definition (PUBLIC CREATE PUBLIC).
        - Attributes, methods, and constructor.
        - Comments for each logic block.
        - Method stubs if implementation not specified.
        - No markdown, only ABAP code.

        Requirement Context:
        {full_context}
        """
        async def run_async():
            return await self._call_llm_with_retry([system_message, HumanMessage(content=draft_prompt)])
        
        draft_code = asyncio.run(run_async())
        # resp_draft = self.llm.invoke([
        #     system_message,
        #     HumanMessage(content=draft_prompt)
        # ])
        # draft_code = getattr(resp_draft, "content", str(resp_draft))
        draft_code = re.sub(r"```(?:abap)?|```", "", draft_code).strip()

        # --- Step 2: Refine Class ---
        refine_prompt = f"""
        Review and refine the following ABAP class code:
        - Ensure global class syntax correctness (CLASS ... DEFINITION/IMPLEMENTATION).
        - Add method documentation comments.
        - Fix indentation and SAP naming standards (lv_, lt_, ls_).
        - Return only the final executable ABAP class code.

        ABAP Class Code:
        {draft_code}
        """

        # resp_refine = self.llm.invoke([
        #     system_message,
        #     HumanMessage(content=refine_prompt)
        # ])
        # final_code = getattr(resp_refine, "content", str(resp_refine))
        # final_code = re.sub(r"```(?:abap)?|```", "", final_code).strip()

        # --- Step 3: Generate Purpose ---
        purpose_prompt = f"""
        You are an SAP documentation specialist.
        Based on the following ABAP class code, generate a concise explanation that includes:
        - The overall purpose of the class.
        - A short description of each method and its role.
        Use structured sentences (no code). Keep it factual and easy to understand.

        ABAP Class Code:
        {draft_code}
        """
        
        resp_purpose = self.llm.invoke([
            SystemMessage(content="You are an SAP documentation writer."),
            HumanMessage(content=purpose_prompt)
        ])
        purpose_text = getattr(resp_purpose, "content", str(draft_code)).strip()

        self.logger.info("✅ Global ABAP class and purpose generated successfully.")
        return {
            "purpose": purpose_text,
            "code": draft_code,
            "type": "class"
        }