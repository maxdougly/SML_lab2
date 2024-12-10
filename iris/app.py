import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Prevent CUDA initialization outside ZeroGPU

import spaces  # Import spaces first
import gradio as gr
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

<<<<<<< HEAD
"""
For more information on `huggingface_hub` Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
"""

def respond(message, history: list[tuple[str, str]], system_message, max_tokens, temperature, min_p,):
=======
# Load the model and tokenizer globally
model = AutoPeftModelForCausalLM.from_pretrained("eforse01/lora_model").to("cuda")  # Move model to CUDA
tokenizer = AutoTokenizer.from_pretrained("eforse01/lora_model")

@spaces.GPU(duration=120)  # Decorate the function for ZeroGPU
def respond(message, history: list[tuple[str, str]], system_message, max_tokens, temperature, min_p):
    # Construct messages for the chat template
>>>>>>> f8b7a6cac70b098afb7034e1d09acc23d05f3c5a
    messages = [{"role": "system", "content": system_message}]
    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})
    messages.append({"role": "user", "content": message})

<<<<<<< HEAD
    model = AutoPeftModelForCausalLM.from_pretrained(
        "eforse01/lora_model", 
    )

    tokenizer = AutoTokenizer.from_pretrained("eforse01/lora_model")
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True, 
        return_tensors = "pt",
    )

    output = model.generate(input_ids = inputs, max_new_tokens = max_tokens,
                    use_cache = True, temperature = temperature, min_p = min_p)
    
    response = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
=======
    # Tokenize the input messages
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",  # Return tensors for PyTorch
    )

    # Ensure input_ids is moved to the same device as the model
    input_ids = inputs.to("cuda")  # Move input_ids to CUDA
    print("Input IDs shape:", input_ids.shape)

    # Generate response
    output = model.generate(
        input_ids=input_ids,  # Pass tensor explicitly as input_ids
        max_new_tokens=max_tokens,
        use_cache=True,
        temperature=temperature,
        min_p=min_p,
    )

    # Debug output
    print("Generated Output Shape:", output.shape)
    print("Generated Output:", output)

    # Decode and format the response
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Yield the response
    yield response.split("assistant")[-1]
>>>>>>> f8b7a6cac70b098afb7034e1d09acc23d05f3c5a

    yield response.split('assistant')[-1]

# Gradio Interface
demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=2048, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=1.5, step=0.1, label="Temperature"),
<<<<<<< HEAD
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.99,
            step=0.01,
            label="Min-p",
        ),
=======
        gr.Slider(minimum=0.1, maximum=1.0, value=0.99, step=0.01, label="Min-p"),
>>>>>>> f8b7a6cac70b098afb7034e1d09acc23d05f3c5a
    ],
)

if __name__ == "__main__":
    demo.launch()