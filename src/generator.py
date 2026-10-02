import os
import time
import warnings
from typing import Optional, List, Dict, Any
from src.config import GEMINI_API_KEY, GEMINI_MODEL_NAME

warnings.filterwarnings("ignore")

AUTOSAR_SYSTEM_PROMPT = """You are an expert Automotive Software Architect specializing in the AUTOSAR (Automotive Open System Architecture) standard (Classic and Adaptive Platforms).

Your goal is to provide accurate, highly technical, and well-structured answers to the user's questions based EXCLUSIVELY on the provided AUTOSAR specification context.

Guidelines:
1. Grounding & Citations: Cite specific requirement IDs (e.g., `[SWS_Can_00017]`), document names, and section/page numbers whenever available.
2. Technical Depth: Detail relevant API signatures, return types, error codes, state machines, and configuration parameters where applicable.
3. Strict Adherence: Do not guess or invent specifications. If the provided context does not contain sufficient details to fully answer the query, clearly state what information is present and what is missing from the indexed specifications.
4. Structure: Use clean Markdown formatting with clear headings, bullet points, and code snippets where appropriate.
"""

QUERY_REFINER_PROMPT = """You are an AUTOSAR technical query optimizer.
Your task is to take a user question and refine it into an optimized technical search query for vector retrieval over AUTOSAR specifications.

Instructions:
1. Expand acronyms and add relevant AUTOSAR domain terms (e.g., MCAL, BSW, RTE, PDU, CanIf, Dem, SOME/IP, Adaptive Platform, Classic Platform).
2. Include potential API names or requirement prefixes (e.g., Can_Write, Dem_SetEventStatus, [SWS_...]).
3. Keep the output focused, dense with technical keywords, and under 30 words.
4. Output ONLY the refined query string with no explanation, quotes, or markdown wrappers.
"""

class GeminiGenerator:
    """Handles LLM response generation and query refinement using Google Gemini."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = GEMINI_MODEL_NAME):
        if api_key is not None:
            self.api_key = api_key
        else:
            self.api_key = GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")

        self.model_name = model_name
        self._client = None
        self._sdk_type = None
        self._init_client()

    def _init_client(self):
        if not self.api_key:
            return

        try:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
            self._sdk_type = "new"
            return
        except ImportError:
            pass

        try:
            import google.generativeai as genai_legacy
            genai_legacy.configure(api_key=self.api_key)
            self._client = genai_legacy
            self._sdk_type = "legacy"
            return
        except ImportError:
            pass

        self._sdk_type = None

    def _call_with_retry(self, model: str, user_content: str, system_prompt: str = AUTOSAR_SYSTEM_PROMPT, max_retries: int = 3) -> str:
        """Calls Gemini with exponential backoff on 503 / 429 errors."""
        for attempt in range(max_retries):
            try:
                if self._sdk_type == "new":
                    from google.genai import types
                    response = self._client.models.generate_content(
                        model=model,
                        contents=user_content,
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            temperature=0.2
                        )
                    )
                    return response.text.strip()
                elif self._sdk_type == "legacy":
                    gen_model = self._client.GenerativeModel(model)
                    full_prompt = f"{system_prompt}\n\n{user_content}"
                    response = gen_model.generate_content(full_prompt)
                    return response.text.strip()
            except Exception as e:
                err_str = str(e)
                if ("503" in err_str or "Service Unavailable" in err_str or "429" in err_str) and attempt < max_retries - 1:
                    sleep_time = (attempt + 1) * 2
                    time.sleep(sleep_time)
                    continue
                raise e

    def refine_query(self, user_query: str) -> str:
        """Refines the user's raw prompt using LLM to generate an optimal technical search query."""
        if not self.api_key or not self._client:
            return user_query  # Fallback to original query if LLM not ready

        candidate_models = [
            self.model_name,
            "gemini-3.8-flash",
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-flash"
        ]

        models_to_try = []
        for m in candidate_models:
            if m and m not in models_to_try:
                models_to_try.append(m)

        for model in models_to_try:
            try:
                refined = self._call_with_retry(
                    model=model,
                    user_content=f"User Query: {user_query}",
                    system_prompt=QUERY_REFINER_PROMPT,
                    max_retries=2
                )
                if refined and len(refined) > 5:
                    return refined
            except Exception:
                continue

        return user_query

    def generate_answer(self, query: str, context: str) -> str:
        """Generate a grounded answer using Gemini LLM given user query and context."""
        if not self.api_key:
            return (
                "⚠️ **GEMINI_API_KEY is not configured.**\n\n"
                "Please set your `GEMINI_API_KEY` in the `.env` file or export it in your terminal:\n"
                "```bash\nexport GEMINI_API_KEY='your_api_key_here'\n```\n\n"
                "### Retrieved Context:\n" + context
            )

        if not self._client:
            self._init_client()
            if not self._client:
                return (
                    "⚠️ **Google GenAI SDK not installed.**\n"
                    "Install it via `pip install google-genai` or `pip install google-generativeai`."
                )

        user_content = f"""User Question:
{query}

Retrieved AUTOSAR Specification Context:
{context}

Please provide a detailed, technical answer addressing the user's question using only the context above. Cite the source files and requirement numbers."""

        candidate_models = [
            self.model_name,
            "gemini-3.8-flash",
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-flash"
        ]

        models_to_try = []
        for m in candidate_models:
            if m and m not in models_to_try:
                models_to_try.append(m)

        last_error = ""
        for model in models_to_try:
            try:
                return self._call_with_retry(model, user_content, system_prompt=AUTOSAR_SYSTEM_PROMPT, max_retries=3)
            except Exception as e:
                last_error = str(e)
                continue

        return f"❌ **Error generating response from Gemini:** {last_error}"
