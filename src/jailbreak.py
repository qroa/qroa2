import os 

from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import numpy as np
import random
from nltk.corpus import words
import nltk
import json

from src.llm_models import get_model
from src.triggers_models import TriggerGenerator, TriggerValidator
from pathlib import Path

import pandas as pd
# Download the word list if not already present
nltk.download("words")


class JailBreak:
    def __init__(self, device, config):
        self.device = device
        self.config = config
        self.model = get_model(
            self.config["model"],
            self.config["auth_token"],
            device,
            self.config["use_system_prompt"],
            self.config["system_prompt"],
            self.config["temperature"],
            self.config["top_p"])
        
        self.model_name = self.config["model"]
        
        self.embedding_model_path = self.config["embedding_model_path"]
        self.reference_embedding, self.tokenizer_surrogate_model = self._load_embedding_model(self.embedding_model_path)
        self.reference_embedding = self.reference_embedding.to(device, dtype=torch.float32)
        self.reference_embedding.requires_grad = False
        
        self.logging_path = os.path.join(self.config['logging_path'], self.model_name)
        self.results_path = os.path.join(self.config['results_path'], self.model_name)

        os.makedirs(self.logging_path, exist_ok=True)
        os.makedirs(self.results_path, exist_ok=True)

        self.logging_generator_path = os.path.join(self.logging_path, "logging_generator.json")
        self.logging_validator_path = os.path.join(self.logging_path, "logging_validator.json")
        
        self.logging_generator_path_csv = os.path.join(self.logging_path, "logging_generator.csv")
        self.logging_validator_path_csv = os.path.join(self.logging_path, "logging_validator.csv")

        self.triggers_path = os.path.join(self.results_path, "triggers.json")
        self.triggers_validate_path = os.path.join(self.results_path, "triggers_validate.json")

    def _load_embedding_model(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer.pad_token_id = tokenizer.unk_token_id

        with torch.no_grad():
            ref_emb = model.get_input_embeddings().weight.data
        del model
        return ref_emb, tokenizer

    def _load_json(self, file_path):
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
        return {}
    
    def run(self, instructions):
        
        print()
        print('Run')
        print('-----------------------')

        triggers = self._load_json(self.triggers_path)
        triggers_validate = self._load_json(self.triggers_validate_path)
    
        if os.path.exists(self.logging_generator_path_csv):
            logging_generator = pd.read_csv(self.logging_generator_path_csv)
        else: 
            logging_generator = pd.DataFrame()

        if os.path.exists(self.logging_validator_path_csv):
            logging_validator = pd.read_csv(self.logging_validator_path_csv)
        else: 
            logging_validator = pd.DataFrame()
    
        for instruction in instructions: 
            trigger_generator = TriggerGenerator(self.model, 
                                                self.device, 
                                                self.config,
                                                self.reference_embedding, 
                                                self.tokenizer_surrogate_model)
            
            trigger_validator = TriggerValidator(self.model, 
                                                self.device, 
                                                self.config)
            
            triggers[instruction] = trigger_generator.run(instruction)

            logging_generator = pd.concat([logging_generator, trigger_generator.return_logging()], ignore_index=True)
            logging_generator.to_json(self.logging_generator_path)
            logging_generator.to_csv(self.logging_generator_path_csv, index=False)
            with open(self.triggers_path, 'w') as f:
                json.dump(triggers, f)

            triggers_validate[instruction] = trigger_validator.run(instruction, triggers[instruction])

            logging_validator = pd.concat([logging_validator, trigger_validator.return_logging()], ignore_index=True)
            logging_validator.to_json(self.logging_validator_path)
            logging_validator.to_csv(self.logging_validator_path_csv, index=False)
            with open(self.triggers_validate_path, 'w') as f:
                json.dump(triggers_validate, f)

        return  triggers, triggers_validate
