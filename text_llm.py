from transformers import AutoModelForCausalLM, AutoTokenizer

# llm_model = AutoModelForCausalLM.from_pretrained("TheBloke/WizardLM-1.0-Uncensored-CodeLlama-34B-GPTQ", code_revision='gptq-4bit-64g-actorder_True', device_map='cuda')
# llm_tokenizer = AutoTokenizer.from_pretrained("TheBloke/WizardLM-1.0-Uncensored-CodeLlama-34B-GPTQ", code_revision='gptq-4bit-64g-actorder_True', device_map='cuda')

def get_reply(text):
    model_inputs = llm_tokenizer([text], return_tensors='pt').to('cuda')
    len_inputs = len(model_inputs['input_ids'])
    generated_ids = llm_model.generate(**model_inputs,
                                        max_new_tokens=200,
                                        do_sample=True,
                                        temperature=0.1,
                                        top_p=0.9,
                                        top_k=20,
                                        repetition_penalty=1.15,
                                       )
    answer_string = llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return answer_string

local_test = 0
if local_test:
    user_text = 'Tell me something about Antarctica.'
    llm_prompt = f'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions.\nUSER: {user_text}'
    llm_prompt = llm_prompt + '\n\nASSISTANT: '
    answer_string = get_reply(llm_prompt)
    print(answer_string)