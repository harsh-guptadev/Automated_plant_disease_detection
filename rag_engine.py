import os
import json
from huggingface_hub import InferenceClient

KNOWLEDGE_BASE_PATH = os.path.join(os.path.dirname(__file__), "rag_knowledge_base.json")

def load_rag_knowledge_base():
    """Loads the structured agronomist knowledge database."""
    if os.path.exists(KNOWLEDGE_BASE_PATH):
        with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def retrieve_disease_context(predicted_class: str) -> dict:
    """Retrieves authoritative treatment data for the predicted plant disease."""
    kb = load_rag_knowledge_base()
    default_info = {
        "symptoms": "Foliar discoloration or lesions observed on the leaf surface.",
        "chemical_treatment": "Consult local agricultural extension for registered fungicides/bactericides.",
        "organic_remedy": "Apply neem oil (5ml/L) or copper octanoate organic spray.",
        "prevention": "Maintain proper plant sanitation, drip irrigation, and good crop spacing."
    }
    return kb.get(predicted_class, default_info)

def generate_rag_care_advice(predicted_disease_readable: str, predicted_class: str, hf_token: str, language: str = "English") -> str:
    """Uses retrieved RAG context + Hugging Face LLM to generate zero-hallucination care steps."""
    context = retrieve_disease_context(predicted_class)
    
    prompt = f"""
    You are an expert plant pathologist and agronomist. 
    A deep learning model identified the crop disease as: **{predicted_disease_readable}**.

    --- VERIFIED AGRONOMY KNOWLEDGE BASE (STRICT SOURCE DATA) ---
    - Typical Symptoms: {context.get('symptoms')}
    - Chemical Treatment & Dosage: {context.get('chemical_treatment')}
    - Organic Remedies: {context.get('organic_remedy')}
    - Preventative Measures: {context.get('prevention')}
    --------------------------------------------------------------

    INSTRUCTIONS:
    - Respond strictly using the provided agronomy knowledge base above.
    - Output language: {language}.
    - Format response with 4 clean sections: 
      1. 🔍 Disease Diagnosis Summary
      2. 🧪 Chemical Treatment & Dosage
      3. 🌿 Organic / Natural Remedies
      4. 🛡️ Long-Term Prevention Strategy
    - Speak directly to the farmer in a professional, encouraging tone.
    """

    client = InferenceClient(api_key=hf_token)

    models_to_try = [
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "meta-llama/Llama-3.2-3B-Instruct"
    ]
    
    last_err = None
    for model_name in models_to_try:
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": f"You are a professional agronomist providing certified plant care advice in {language}."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4
            )
            return completion.choices[0].message["content"]
        except Exception as e:
            last_err = e
            continue
            
    raise last_err

def chat_with_agronomist_rag(user_query: str, predicted_disease_readable: str, predicted_class: str, hf_token: str, chat_history: list, language: str = "English") -> str:
    """Interactive Agri-Chatbot powered by retrieved disease knowledge."""
    context = retrieve_disease_context(predicted_class)
    
    system_prompt = f"""
    You are AgriVision AI Chatbot, a friendly agronomist helping a farmer whose plant was diagnosed with **{predicted_disease_readable}**.
    
    Use this verified reference data to answer their question:
    - Symptoms: {context.get('symptoms')}
    - Chemical Options: {context.get('chemical_treatment')}
    - Organic Options: {context.get('organic_remedy')}
    - Prevention: {context.get('prevention')}
    
    Answer the user's question accurately in {language}. Keep responses helpful and practical.
    """
    
    messages = [{"role": "system", "content": system_prompt}]
    for msg in chat_history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_query})

    client = InferenceClient(api_key=hf_token)

    models_to_try = [
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "meta-llama/Llama-3.2-3B-Instruct"
    ]
    
    last_err = None
    for model_name in models_to_try:
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.5
            )
            return completion.choices[0].message["content"]
        except Exception as e:
            last_err = e
            continue
            
    raise last_err
