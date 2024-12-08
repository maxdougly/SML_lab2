#### Overview

This lab focuses on fine-tuning a Large Language Model (LLM) with limited computational resources and deploying it as a conversational chatbot. Using the FineTome-100k dataset and the PEFT technique with LoRA, the model is trained efficiently for better performance without extensive GPU requirements. The process also leverages the **Unsloth framework** for memory-efficient fine-tuning and inference acceleration. Finally, the fine-tuned model is integrated into a user-friendly Gradio interface and hosted on Hugging Face Spaces for public interaction.

#### Main Steps

1. **Fine-Tune the Model**:
   - Select a pre-trained LLM (Llama-3 1B from Hugging Face).
   - Use Parameter Efficient Fine-Tuning (PEFT) with LoRA for resource-efficient updates.
   - Train on the *FineTome-100k dataset*, an open-source instruction-tuning dataset designed for multi-turn conversational fine-tuning of large language models.
   - Periodically save and checkpoint for model weights using Unsloth to optimize memory and speed up training.

2. **Save the Model**:
   - Store the fine-tuned model on Hugging Face for easy integration into applications.

3. **Build a User Interface**:
   - Create a Gradio-based chatbot UI.
   - Deploy the app on Hugging Face Spaces.

#### Sources
- **Huggingface public URL:** [Gradio Deployment](https://huggingface.co/spaces/maxdougly/iris)  
- **FineTome-100k dataset:** [Dataset Link](https://huggingface.co/datasets/mlabonne/FineTome-100k)  
- **Unsloth Framework:** [GitHub Repository](https://github.com/unslothai/unsloth)
