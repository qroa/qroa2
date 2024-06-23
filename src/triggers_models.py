import json
import os
import time
import random
from tqdm import tqdm
import torch


import numpy as np
import random
from nltk.corpus import words
import nltk

from src.score_function import scoring_function_factory
from src.qroa_models import SurrogateModel, AcquisitionFunction
from src.loss_functions import  MSELoss
from pathlib import Path
import scipy.stats as stats

import pandas as pd
# Download the word list if not already present
nltk.download("words")


class TriggerGenerator:
    """A class for generating triggers using a surrogate model."""

    def __init__(self, 
                 model, 
                 device, 
                 config,
                 reference_embedding,
                 tokenizer_surrogate_model
                 ):
        self.model = model
        self.device = device
        self.config = config

        self.coordinates_length = self.config["len_coordinates"]
        self.learning_rate = self.config["learning_rate"]
        self.weight_decay = self.config["weight_decay"]
        self.nb_epochs = self.config["nb_epochs"]
        self.scoring_type = self.config["scoring_type"]
        self.max_generations_tokens = self.config["max_generations_tokens"]
        self.batch_size = self.config["batch_size"]
        self.topk = self.config["topk"]
        self.max_d = self.config["max_d"]
        self.ucb_c = self.config['ucb_c']
        self.triggers_init = self.config['triggers_init']
        self.threshold = self.config['threshold']
        self.reference_embedding = reference_embedding
        self.tokenizer_surrogate_model = tokenizer_surrogate_model

        self.scoring_function = scoring_function_factory(self.scoring_type, self.device)

        self.token_count = self.reference_embedding.shape[0]

        self.surrogate_model = SurrogateModel(self.coordinates_length, self.reference_embedding).to(self.device)

        self.acquisition_function = AcquisitionFunction(
            self.token_count,   
            self.coordinates_length,
            self.device,
            self.tokenizer_surrogate_model,
        )

        self.opt1 = torch.optim.Adam(
            self.surrogate_model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        self.learning_loss = MSELoss()
        self.coordinates = list(range(self.coordinates_length))
        self.word_list = self.tokenizer_surrogate_model.batch_decode(
            list(self.tokenizer_surrogate_model.vocab.values())
        )

        self.surrogate_model.train()

        self.D = []
        self.best_triggers = set()
        self.n = dict()
        self.h = dict()
        self.N = 0
        self.logging = pd.DataFrame()
        self.loss = 0

    def _optimization_step(self):
        """Performs an optimization step in the training of the surrogate model."""
        
        size_sample = min([len(self.D), self.batch_size])
        idx = np.random.choice(len(self.D), size=size_sample, replace=False)
        samples_inputs = [self.D[i] for i in idx]
        x = self._encode(samples_inputs)
        x = x.to(self.device)

        self.opt1.zero_grad()
        value_estimates = self.surrogate_model(x).view(-1)
        y = torch.tensor([self.h[i] for i in samples_inputs], device=self.device).view(-1 )
        loss = self.learning_loss(value_estimates, y)
        loss.backward()
        self.opt1.step()

        return loss

    def _encode(self, input_strings):
        """Encodes a string using the black box tokenizer."""
        return self.tokenizer_surrogate_model(
            input_strings,
            return_tensors="pt",
            max_length=self.coordinates_length,
            padding="max_length",
            add_special_tokens=False,
            truncation=True,
        ).to(self.device)['input_ids']

    def _update_memory(self, triggers, scores):

        for idx, trigger_test in enumerate(triggers):
            z = trigger_test
            s_z = scores[idx].item()

            if trigger_test in self.h:
                self.h[z] = (self.n[z] * self.h[z] + s_z)/(self.n[z]+1)
                self.n[z] = self.n[z] + 1
            else: 
                self.h[z] = s_z
                self.n[z] =  1
        
            self.D += triggers
            while len(self.D) > self.max_d:
                self.D.pop(0)

    def _eval_triggers(self, instruction, triggers):

        instructions = [instruction for _ in triggers]
        prompts = [instruction + t for t in triggers]
        generations = self.model.generate(prompts, max_tokens=self.max_generations_tokens)
        score_array = self.scoring_function.score(instructions, generations)

        return score_array

    def _add_logging(self, 
                     instruction, 
                     trigger, 
                     epoch):

        logging_json = {'model_name': self.config["model"],
                        'embedding_model_name': self.config["embedding_model_path"],
                        'scoring_method': self.config["scoring_type"],
                        'nb_tokens': self.coordinates_length,
                        'batch_size': self.batch_size,
                        'topk' : self.config["topk"],
                        'max_d' : self.config["max_d"],
                        'ucb_c' : self.config['ucb_c'],
                        'trigger': trigger,
                        'instruction': instruction,
                        'average_score': self.h[trigger],
                        'nb_test': self.n[trigger],
                        'epoch': epoch,
                        'budget': self.N,
                        'loss': self.loss.item()}  

        df_dictionary = pd.DataFrame([logging_json])
        self.logging = pd.concat([self.logging, df_dictionary], ignore_index=True)

    def return_logging(self):
        
        return self.logging 
    
    def _generate_triggers(self, instruction):

        """Generates a single trigger for the given target."""
                        
        self.triggers_init += ["".join(random.choice(self.word_list) for _ in range(self.coordinates_length * 5))]
        trigger_ids = self._encode(self.triggers_init)
        triggers = self.tokenizer_surrogate_model.batch_decode(trigger_ids)

        score_array = self._eval_triggers(instruction, triggers)
        self._update_memory(triggers, score_array)

        with tqdm(
            range(self.nb_epochs),
            desc=f"Best Score: Unkown Loss: Unkown",
            unit="epoch",
        ) as progress_bar:
            for current_epoch in progress_bar:
                
                if current_epoch % len(self.coordinates) == 0:
                    random.shuffle(self.coordinates)

                # Selection Phase 
                with torch.no_grad():
                    
                    self.N =  sum([self.n[key] for key in self.n.keys()])
                    ucb_b = {trigger: self.h[trigger] + self.ucb_c*np.sqrt(np.log(self.N)/(self.n[trigger]+1)) for trigger in self.h}
                    trigger = max(self.h, key=lambda key: ucb_b[key])
                    current_coordinate = self.coordinates[current_epoch % self.coordinates_length]
                    top_k_triggers = self.acquisition_function(self.surrogate_model, trigger, current_coordinate, self.topk)

                    # s = self._eval_triggers(instruction, [trigger]*self.config["nb_samples_per_trigger"])
                    # print(s.mean().item())

                    # Eval Phase Phase 
                    score_array = self._eval_triggers(instruction, top_k_triggers)
                    self._update_memory(top_k_triggers, score_array)
                    max_n = max([self.n[key] for key in self.n.keys()])

                #Learning phase
                self.loss = self._optimization_step()
                if self.h[trigger] >= self.threshold:
                    self.best_triggers.add(trigger)

                self._add_logging(instruction,
                                  trigger,
                                  current_epoch)

                progress_bar.set_description(f"Trigger Score: {self.h[trigger]}, Loss: {self.loss:.4f}, Max n: {max_n}")
            
            return list(self.best_triggers)


    def run(self, instruction):
        """Generates multiple triggers for the given instruction."""
        print(f"Generate triggers for instruction: {instruction}")
        triggers = self._generate_triggers(instruction)

        return triggers


class TriggerValidator:
    """A class for generating triggers using a surrogate model."""

    def __init__(self, 
                 model, 
                 device, 
                 config):
        
        self.model = model
        self.device = device
        self.config = config
        self.scoring_type = self.config["scoring_type"]
        self.nb_samples = self.config["nb_samples_per_trigger"]
        self.threshold = self.config["threshold"]
        self.scoring_function = scoring_function_factory(self.scoring_type, self.device)
        self.max_generations_tokens = self.config["max_generations_tokens"]
        self.p_value = self.config["p_value"]
        self.logging = pd.DataFrame()

    def _eval_triggers(self, instruction, triggers):

        instructions = [instruction for _ in triggers]
        prompts = [instruction + t for t in triggers]
        generations = self.model.generate(prompts, max_tokens=self.max_generations_tokens)
        score_array = self.scoring_function.score(instructions, generations)

        return score_array

    def _add_logging(self, 
                     instruction, 
                     trigger, 
                     mean, 
                     std,
                     z):

        logging_json = {'model_name': self.config["model"],
                        'scoring_method': self.config["scoring_type"],
                        'trigger': trigger,
                        'instruction': instruction,
                        'mean': mean,
                        'std': std,
                        'z': z,
                        'nb_samples': self.nb_samples}  
        
        df_dictionary = pd.DataFrame([logging_json])
        self.logging = pd.concat([self.logging, df_dictionary], ignore_index=True)

    def return_logging(self):
        
        return self.logging 
    
    def validate(self, instruction, triggers):

            
        triggers_validated = set()
        
        for t in tqdm(triggers): 
        
            with torch.no_grad():
                resampled_triggers = [t]*self.nb_samples
                score_array = self._eval_triggers(instruction, resampled_triggers)

                mean = score_array.mean().item()
                std = score_array.std().item()
                z = (mean - self.threshold)/(std/np.sqrt(self.nb_samples))
                z_critical = stats.norm.ppf(1-self.p_value) 

            self._add_logging(instruction, 
                              t, 
                              mean, 
                              std,
                              z)

            if z>=z_critical:
                triggers_validated.add(t)

        return list(triggers_validated)


    def run(self, instruction, triggers):

        print(f"Validate triggers for instruction: {instruction}")

        triggers_validated = self.validate(instruction, triggers)

        return triggers_validated


