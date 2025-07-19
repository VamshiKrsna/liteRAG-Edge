import os 
import psutil 
import time 
import csv 
import gc 
from transformers import AutoTokenizer, AutoModelForCausalLM

MODELS = [
    ("meta-llama/Meta-Llama-3-8B", "Llama 3 8B"),
    ("meta-llama/Meta-Llama-3-3B", "Llama 3 3B"),
    ("meta-llama/Meta-Llama-3-1B", "Llama 3 1B"),
    ("microsoft/phi-3-mini-4k-instruct", "Phi 3.5 Mini 3.8B"),
    ("Qwen/Qwen1.5-0.5B-Chat", "Qwen2 0.5B"),
    ("Qwen/Qwen1.5-1B-Chat", "Qwen2 1B"),
    ("Qwen/Qwen1.5-7B-Chat", "Qwen2 7B"),
    ("mistralai/Mistral-7B-Instruct-v0.2", "Mistral 7B"),
    ("mistralai/Mistral-7B-v0.1", "Mistral Small 7B"),  
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama 1.1B"),
    ("google/gemma-2b-it", "Gemma 2B"),
    ("google/gemma-7b-it", "Gemma 7B"),
    ("google/gemma-9b", "Gemma 9B"),
]

BENCHMARK_CSV_PATH = "llm_benchmark_results.csv"
PROMPT = "Explain AI in a single sentence."

def get_memory(): 
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024**2) 

def benchmark(model_id, model_desc):
    print(f"Testing {model_desc}")
    mem_start = get_memory()
    t_load_start = time.time() 
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if "qwen" in model_id.lower() or "mistral" in model_id.lower():
            # qwen and mistral models need trust_remote_code = True
            model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
        else: 
            model = AutoModelForCausalLM.from_pretrained(model_id)
    except Exception as e:
        print(f"{model_desc} failed to load: {e}")
        return [model_desc, "load error", e]
    t_load_time = time.time() - t_load_start
    mem_after_load = get_memory()
    try : 
        t_inf_start = time.time() 
        inputs = tokenizer(PROMPT, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=32)
        t_inf_time = time.time() - t_inf_start
        mem_after_inf = get_memory()
        reply = tokenizer.decode(out[0], skip_special_tokens = True)
        result = [
            model_desc, 
            f"{t_load_time:.2f}",
            f"{max(mem_start, mem_after_load, mem_after_inf):.2f}",
            f"{t_inf_time:.2f}", 
            reply[:60].replace('\n',' ')
        ]
    except Exception as e:
        print(f"{model_desc} failed to infer: {e}")
        return [model_desc, "inference error", e]
    del model 
    del tokenizer
    gc.collect() 
    return result 

def main():
    header = ["Model", "Load Time (s)", "Peak RAM (MB)", "Inference Time (s)", "Output Sample"]
    if not os.path.exists(BENCHMARK_CSV_PATH):
        with open(BENCHMARK_CSV_PATH, "w", newline="") as f: 
            csv.writer(f).writerow(header)
    for model_id, model_desc in MODELS: 
        row = benchmark(model_id, model_desc) 
        with open(BENCHMARK_CSV_PATH, "a", newline="") as f: 
            csv.writer(f).writerow(row) 

if __name__ == "__main__": 
    main() 

