import numpy as np
import requests
import streamlit as st
import openai

def main():
    st.title("Scientific Question Generation")

    st.write("This application is designed to generate a question given a piece of scientific text.\
    We include the output from four different models, the [BART-Large](https://huggingface.co/dhmeltzer/bart-large_askscience-qg) and [FLAN-T5-Base](https://huggingface.co/dhmeltzer/flan-t5-base_askscience-qg) models \
    fine-tuned on the r/AskScience split of the [ELI5 dataset](https://huggingface.co/datasets/eli5) as well as the zero-shot output \
    of the [FLAN-T5-XXL](https://huggingface.co/google/flan-t5-xxl) model and the [GPT-3.5-turbo](https://platform.openai.com/docs/models/gpt-3-5) model.\
    \n\n For a more thorough discussion of question generation see this [report](https://wandb.ai/dmeltzer/Question_Generation/reports/Exploratory-Data-Analysis-for-r-AskScience--Vmlldzo0MjQwODg1?accessToken=fndbu2ar26mlbzqdphvb819847qqth2bxyi4hqhugbnv97607mj01qc7ed35v6w8) for EDA on the r/AskScience dataset and this \
    [report](https://api.wandb.ai/links/dmeltzer/7an677es) for details on our training procedure.\
    \n\n**Disclaimer**: You may recieve an error message when you first run the model. We are using the Huggingface API to access the BART-Large and FLAN-T5 models, and the inference API takes around 20 seconds to load the model.\
    In addition, the FLAN-T5-XXL model was recently updated on Huggingface and may give buggy outputs.\
    ")
    
    checkpoints = ['dhmeltzer/bart-large_askscience-qg',
              'dhmeltzer/flan-t5-base_askscience-qg',
              'google/flan-t5-xxl']
    
    headers = {"Authorization": f"Bearer {st.secrets['HF_token']}"}
    openai.api_key = st.secrets['OpenAI_token']
    
    def query(checkpoint, payload):
        API_URL = f"https://api-inference.huggingface.co/models/{checkpoint}"
        
        response = requests.post(API_URL, 
                                    headers=headers, 
                                    json=payload)
        
        return response.json()
    
    # User search
    user_input = st.text_area("Question Generator", 
                                """Black holes are the most gravitationally dense objects in the universe.""")
    
    if user_input:
        for checkpoint in checkpoints:
            
            model_name = checkpoint.split('/')[1]
    
            if 'flan' in model_name.lower():
                
                prompt = 'generate a question: ' + user_input

            else:
                prompt = user_input
            
            output = query(checkpoint,{
                        "inputs": prompt,
                        "wait_for_model":True})
            try:
                output=output[0]['generated_text']
            except:
                st.write(output)
                return
            
            st.write(f'**{model_name}**: {output}')
    
        model_engine = "gpt-3.5-turbo"
        max_tokens = 50
        
        prompt = f"generate a question: {user_input}"
    
        response=openai.ChatCompletion.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates questions from text."},
                {"role": "user", "content": prompt},
            ])
    
        output = response['choices'][0]['message']['content']
        
        st.write(f'**{model_engine}**: {output}')

        
if __name__ == "__main__":
    main()
#[0]['generated_text']