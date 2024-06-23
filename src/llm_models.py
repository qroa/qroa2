from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from vllm import LLM, SamplingParams
from huggingface_hub import login
from collections import defaultdict
from fastchat.conversation import get_conv_template
from openai import OpenAI

from abc import ABC, abstractmethod
import concurrent.futures

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage


class Model(ABC):

    def __init__(self, 
                 auth_token, 
                 device, 
                 use_system_prompt, 
                 system_prompt):
        
        self.auth_token = auth_token
        self.device = device
        self.use_system_prompt = use_system_prompt
        self.system_prompt = system_prompt
        self.template_name = None

    def process_prompts(self, prompts):
        input_prompts = []
        for text in prompts:
            if self.template_name == "llama-2":
                system_template = f"<s><s>[INST] <<SYS>>\n{self.system_prompt}\n<</SYS>>\n\n{text}[/INST]"
                input_prompts.append(system_template) 
            else: 
                conv = get_conv_template(self.template_name)
                conv.set_system_message(self.system_prompt)
                conv.append_message(conv.roles[0], text)
                conv.append_message(conv.roles[1], None)
                input_prompts.append(conv.get_prompt())

        return input_prompts

    @abstractmethod 
    def generate(self, prompt, max_tokens):
        """
        Generate a response based on the prompt.
        This method needs to be implemented by subclasses.
        """
        pass



class HuggingFaceModel(Model):
    model_details = {
        "llama2_chat_hf": ("meta-llama/Llama-2-7b-chat-hf", "llama-2"),
        "llama2_hf": ("meta-llama/Llama-2-7b-hf", "llama-2"),
        "vicuna_hf": ("lmsys/vicuna-7b-v1.3", "vicuna_v1.1"),
        "mistral_hf": ("mistralai/Mistral-7B-Instruct-v0.1", "mistral"),
        "falcon_hf": ("tiiuae/falcon-7b-instruct", "falcon"),
        "mpt_hf": ("mosaicml/mpt-7b-instruct", "mpt-7b-chat"),
    }

    def __init__(self, 
                 auth_token, 
                 device, 
                 use_system_prompt, 
                 system_prompt,
                 model_name,
                 temperature,
                 top_p):
        
        super().__init__(auth_token, 
                         device, 
                         use_system_prompt, 
                         system_prompt)

        path, template_name = self.model_details[model_name]
        if auth_token is not None:  
            login(token=self.auth_token, add_to_git_credential=True)

        self.template_name = template_name
        self.path = path
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p

        self.tokenizer = AutoTokenizer.from_pretrained(self.path, 
                                                       padding_side="left",
                                                       trust_remote_code=True)
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.path,
                torch_dtype=torch.bfloat16,
                device_map="balanced",
                trust_remote_code=True,
            )
            .eval()
            .to(device)
        )

        if 'llama-2' in model_name:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompts, max_tokens):
        
        if isinstance(prompts, str):
            prompts = [prompts]

        if self.use_system_prompt:
            prompts = self.process_prompts(prompts)

        tokenization = self.tokenizer(prompts, padding=True, return_tensors="pt")
        tokenization["input_ids"] = tokenization["input_ids"].to(self.device)
        tokenization["attention_mask"] = tokenization["attention_mask"].to(self.device)
        tokenization.update(
            {
                "max_new_tokens": max_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
                "do_sample": True,
                "temperature": self.temperature,
                "top_p": self.top_p,
            }
        )

        outputs = self.model.generate(**tokenization)
        generations = self.tokenizer.batch_decode(
            outputs[:, -max_tokens:], skip_special_tokens=True
        )

        return generations


class MistralModel(Model):

    def __init__(
        self, auth_token, device, system_prompt
    ):
        super().__init__(auth_token, device, True, system_prompt)

        self.client = MistralClient(api_key=auth_token)
        self.model_name = "mistral-small-latest"

    def generate(self, prompts, max_tokens):
        if isinstance(prompts, str):
            prompts = [prompts]

        prompts = [[ChatMessage(role="user", content=prompt)] for prompt in prompts]

        def fetch_generation(prompt):
            output = self.client.chat(
                model=self.model_name,
                messages=prompt,
                max_tokens=max_tokens
            )
            return output.choices[0].message.content

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(fetch_generation, prompt): i for i, prompt in enumerate(prompts)}
            results = [None] * len(prompts)
            for future in concurrent.futures.as_completed(futures):
                result_position = futures[future]
                results[result_position] = future.result()

        return results
    
    
class OpenaiModel(Model):

    def __init__(
        self, auth_token, device, system_prompt
    ):
        super().__init__(auth_token, device, True, system_prompt)

        self.client = OpenAI(api_key=auth_token)
        self.model_name = "gpt-3.5-turbo-0613"

    def generate(self, prompts, max_tokens):
        if isinstance(prompts, str):
            prompts = [prompts]

        prompts = [
            [{"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}]
            for prompt in prompts
        ]

        def fetch_generation(prompt):
            output = self.client.chat.completions.create(
                model=self.model_name,
                messages=prompt,
                max_tokens=max_tokens
            )
            return output.choices[0].message.content

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(fetch_generation, prompt): i for i, prompt in enumerate(prompts)}
            results = [None] * len(prompts)
            for future in concurrent.futures.as_completed(futures):
                result_position = futures[future]
                results[result_position] = future.result()

        return results

# Factory Method
def get_model(model_name, 
              auth_token, 
              device, 
              use_system_prompt,
              system_prompt,
              temperature,
              top_p):
    
    if model_name.lower() in HuggingFaceModel.model_details:
        return HuggingFaceModel(
            auth_token, 
            device, 
            use_system_prompt, 
            system_prompt, 
            model_name.lower(),
            temperature,
            top_p
        )
    elif model_name.lower() == 'openai':
        return OpenaiModel(
            auth_token, device, system_prompt
        )
    elif model_name.lower() == 'mistral':
        return MistralModel(
            auth_token, device, system_prompt
        )
    else:
        raise ValueError(f"Unknown model type: {model_name}")
