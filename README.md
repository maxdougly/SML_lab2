#### Overview

This lab focuses on fine-tuning a Large Language Model (LLM) with limited computational resources and deploying it as a conversational chatbot. 

Using the *FineTome-100k dataset* and the PEFT technique with LoRA, the model is trained efficiently for better performance without extensive GPU requirements. The process also leverages the **Unsloth framework** for memory-efficient fine-tuning and inference acceleration. Finally, the fine-tuned model is integrated into a Gradio interface and hosted on Hugging Face Spaces for public interaction.

#### Task 1

1. **Fine-Tune a Model**:
   - Select a pre-trained LLM (Llama-3 1B from Hugging Face).
   - Use Parameter Efficient Fine-Tuning (PEFT) with LoRA for resource-efficient updates.
   - Train on the *FineTome-100k dataset* open-source instruction-tuning dataset designed for multi-turn conversational fine-tuning of large language models.
   - Train for one epoch and periodically save and checkpoints for model weights every 1,250th step.

2. **Save the Model on Hugging Face**:
   - Once the model is trainded, store it on Hugging Face for integration into user interface.

3. **Build a serverless UI for using the model**:
   - Create a Gradio-based chatbot UI.
   - Deploy the app on Hugging Face Spaces.

##### Result: 
**Fine-tuned Model:** https://huggingface.co/eforse01/lora_model
**Huggingface public URL:** https://huggingface.co/spaces/maxdougly/iris

#### Task 2

Model performance can be improved several ways, such as:

a) **Model-Centric Approach:** 
   - **Hyperparameter tuning:** Conduct grid search to systematically tune parameters like learning rate, batch size, and dropout. Use techniques such as Adam optimizer with weight decay for efficient updates.
   - **Model Size:** Experiment with larger models like Llama-3B instead of smaller variants, leveraging their increased capacity for better understanding complex patterns in data.
   - **Efficient Resource Utilization:** Focus on optimizing computational and memory efficiency to enable the training of larger and more complex models without exceeding hardware limitations.

b) **Data-Centric Approach:** 
   - **Choosing another dataset:** Incorporate datasets from similar domains or different instruction-based datasets to improve task relevance and domain adaptation.  
   - **Dataset Size:** Combine datasets from multiple sources, such as Hugging Face FineTome-100k and domain-specific datasets, to increase coverage and reduce task-specific biases.
   - **Class Balance:** Analyse class distributions and oversample underrepresented categories to ensure a balanced dataset, reducing model bias.
   - **Data Augmentation:** Use techniques like back-translation, paraphrasing, and synthetic data generation to diversify the dataset and improve robustness.

#### Other sources
- **FineTome-100k dataset:** https://huggingface.co/datasets/mlabonne/FineTome-100k
- **Unsloth Framework:** https://github.com/unslothai/unsloth 
