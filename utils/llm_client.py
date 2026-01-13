import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# Hugging Face LLaMA setup
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

if HF_AVAILABLE:
    model_name = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # ← Fixed this line!
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    llama_pipeline = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer,
        device=0  # Use GPU
    )

# OpenAI setup
if OPENAI_KEY:
    import openai
    openai_client = openai.OpenAI(api_key=OPENAI_KEY)


def chat(messages, model_name="gpt-3.5-turbo", max_tokens=150):
    """
    Unified chat interface for MCP server/client.
    Uses OpenAI if API key exists, else falls back to LLaMA.
    """
    if OPENAI_KEY:
        try:
            response = openai_client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except openai.RateLimitError:
            print("OpenAI quota exceeded, falling back to LLaMA...")
        except Exception as e:
            print(f"OpenAI error: {e}, falling back to LLaMA...")

    if HF_AVAILABLE:
        # Format prompt better for LLaMA
        prompt = ""
        system_msg = ""
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                system_msg = content
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
        
        # Add system message at the beginning if exists
        if system_msg:
            prompt = f"System: {system_msg}\n\n{prompt}"
        
        prompt += "Assistant:"
        
        # Generate with better parameters
        output = llama_pipeline(
            prompt,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            num_return_sequences=1
        )[0]["generated_text"]
        
        # Extract only the new generated text after "Assistant:"
        response = output.split("Assistant:")[-1].strip()
        
        # Stop at newlines or common stop patterns
        stop_patterns = ["\nUser:", "\nSystem:", "\n\n"]
        for pattern in stop_patterns:
            if pattern in response:
                response = response.split(pattern)[0].strip()
        
        return response
    else:
        return "No LLM available: install transformers or set OPENAI_API_KEY."