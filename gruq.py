import re
from dotenv import load_dotenv
load_dotenv()

import os
import openai
import dspy
from functools import cache, lru_cache
import json
from openai import OpenAI
import hashlib
import yaml
import requests
from bs4 import BeautifulSoup

import numpy as np
from random import shuffle

from dspy.signatures.signature import signature_to_template
from dspy.evaluate import Evaluate


class BuildMessages:
    def __init__(self, system_prompt, user_prompt):
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

    def render(self, **kwargs):
        sys = self.system_prompt.format(**kwargs)
        user = self.user_prompt.format(**kwargs)

        return [
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ]

class GrugTranslation(dspy.Signature):
    "Translate plain english to grug text"
    plain_english = dspy.InputField()
    grug_text = dspy.OutputField()

class CoT(dspy.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prog = dspy.ChainOfThought(GrugTranslation)

    def forward(self, plain_english):
        return self.prog(plain_english=plain_english)

class Predict(dspy.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prog = dspy.Predict(GrugTranslation)

    def forward(self, plain_english):
        return self.prog(plain_english=plain_english)

# https://dspy-docs.vercel.app/docs/building-blocks/metrics#intermediate-using-ai-feedback-for-your-metric
class AssessBasedOnQuestion(dspy.Signature):
    """Given the assessed text, explain and then provide a yes or no to the assessment question"""

    assessed_text = dspy.InputField(format=str)
    assessment_question = dspy.InputField(format=str)
    assessment_explanation = dspy.OutputField(desc="Describe in 3 bullet points why the text are similar or not")
    assessment_answer = dspy.OutputField(desc="Yes, No or Kind of")

## Metrics

def automated_readability_index(text):
    import re 

    # Add a newline to text if it doesn't end in a sentence
    if not text.strip().endswith(('.', '!', '?', '\n')):
        text += '\n'

    characters = len(re.sub(r'\s+', '', text))
    words = len(text.split())

    sentences = len(re.findall(r'[.!?\n]', text))

    if words == 0 or sentences == 0:
        return 0
    
    ari = (4.71 * (characters / words)) + (0.5 * (words / sentences)) - 21.43

    return round(ari, 2)

gpt4o = dspy.OpenAI(model="gpt-4o", max_tokens=1000)

def similarity_metric(truth, pred, trace=None):
    truth_grug_text = truth.grug_text
    proposed_grug_text = pred.grug_text
    similarity_question = f"""
    Is the proposed grug text similar in meaning to the golden_standard truth text? 
    
    Golden Standard: {truth_grug_text}. 

    Answer only with yes, no or kind of.
    """

    with dspy.context(lm=gpt4o):
        assessor = dspy.Predict(AssessBasedOnQuestion)
        res = assessor(assessed_text=proposed_grug_text, assessment_question=similarity_question)
    print(res)
    raw_similarity = res.assessment_answer.lower().strip()
    return (res.assessment_answer == "yes") or (res.assessment_answer == "kind of")

def ari_metric(truth, pred, trace=None):
    truth_ari = automated_readability_index(truth.grug_text)
    proposed_ari = automated_readability_index(pred.grug_text)
    res = proposed_ari <= 7.01
    return res

def overall_metric(truth, pred, trace=None):
    similarity = similarity_metric(truth, pred, trace)
    ari = ari_metric(truth, pred, trace)
    return similarity and ari

class GrugTranslator:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI()
        self.openai_model_name = "gpt-4o-mini"
        self.cache_dir = os.path.join(os.path.dirname(__file__), '.cache')
        os.makedirs(self.cache_dir, exist_ok=True)

    def disk_cache(self, func):
        cache_file = os.path.join(self.cache_dir, f'{func.__name__}_cache.json')

        def load_cache():
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    # print(f"Loading cache from {cache_file}")
                    return json.load(f)
            return {}

        def save_cache(cache):
            with open(cache_file, 'w') as f:
                json.dump(cache, f)
                print(f"Saved cache to {cache_file}")

        cache = load_cache()

        @lru_cache(maxsize=None)
        def wrapper(*args, **kwargs):
            # Create a hash of the arguments
            key = hashlib.md5(str(args).encode() + str(kwargs).encode()).hexdigest()
            if key not in cache:
                cache[key] = func(*args, **kwargs)
                save_cache(cache)
            return cache[key]

        return wrapper

    @property
    def translate_grug(self):
        @self.disk_cache
        def translate(grug_text):
            prompt = BuildMessages(
                "You are an expert in deciphering strange text. The user will provide text written by someone named Grug and you will provide the translation.",
                """Translate the following text into plain english: '{text}'. 
                
                Do not respond with any other text. Only provide that text. Now take a deep breath and begin."""
            )
            result = self.client.chat.completions.create(
                model=self.openai_model_name,
                messages=prompt.render(text=grug_text), # type: ignore
            )
            return result.choices[0].message.content
        return translate

    def initialize_grug_text_from_web(self):
        url = "https://grugbrain.dev/"
        res = requests.get(url)
        soup = BeautifulSoup(res.text, 'html.parser')
        raw_text = [p.text for p in soup.find_all('p') if p.text]
        
        file_path = os.path.expanduser("~/code/others/llms/dspy-grug/data/raw-text.txt")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, "w") as f:
            f.write("\n".join(raw_text))
        
        print(f"Grug text initialized and saved to {file_path}")
        return raw_text

    def read_grug_text(self):
        file_path = os.path.expanduser("~/code/others/llms/dspy-grug/data/raw-text.txt")
        if not os.path.exists(file_path):
            print("Raw text file not found. Initializing from web...")
            return self.initialize_grug_text_from_web()
        
        with open(file_path, "r") as f:
            return f.read().splitlines()

    def create_dataset(self):
        raw_text = self.read_grug_text()
        dataset = []
        for grug_text in raw_text[:10]:
            dataset.append({
                "grug_text": grug_text,
                "translation": self.translate_grug(grug_text),
            })
        return dataset

    def create_examples(self, dataset):
        examples = []
        for row in dataset:
            examples.append(
                dspy.Example(
                    grug_text=row["grug_text"],
                    plain_english=row["translation"],   
                ).with_inputs("plain_english")
            )
        return examples

    def split_for_train_test(self, values, test_size = 1/3.0):
        shuffle(values)
        split_index = int(len(values) * test_size)
        return values[:split_index], values[split_index:]

    def pretty_print_examples(self, examples, title=None):
        if title:
            print(f"\n{title}")
        for i, example in enumerate(examples, 1):
            example_dict = {
                "grug_text": example.grug_text,
                "plain_english": example.plain_english
            }
            yaml_str = yaml.dump({f"Example {i}": example_dict}, default_flow_style=False)
            print(yaml_str)

    def setup_dspy_translation(self):
        grug_translation_as_template = signature_to_template(GrugTranslation)
        return GrugTranslation, grug_translation_as_template

    def test_cot(self):
        c = CoT()
        print(c.forward("You should not construct complex systems."))

    def test_predict(self):
        p = Predict()
        print(p.forward("You should not construct complex systems."))

    def test_ari(self):
        for ex in self.create_examples(self.create_dataset()):
            grug_ari = automated_readability_index(ex.grug_text)
            source_ari = automated_readability_index(ex.plain_english)

            print(f"Grug Text: {ex.grug_text}")
            print(f"Plain English: {ex.plain_english}")
            print(f"ARI: {source_ari} -> {grug_ari}")
            print("\n")

    def test_assess(self):
        example = dspy.Example(assessed_text="This is a test.", assessment_question="Is this a test?", assessment_answer="Yes").with_inputs("assessed_text", "assessment_question")
        res = signature_to_template(AssessBasedOnQuestion).query(example)
        print(res)

    def optimize_model(self, train, test):
        from dspy.teleprompt import BootstrapFewShot
        # Print code location of dspy.teleprompt and BootstrapFewShot
        import inspect
        import dspy.teleprompt
        
        print(f"dspy.teleprompt location: {inspect.getfile(dspy.teleprompt)}")
        print(f"BootstrapFewShot location: {inspect.getfile(BootstrapFewShot)}")

        config = dict(max_bootstrapped_demos=4, max_labeled_demos=4)
        optimizer = BootstrapFewShot(metric=overall_metric, **config)
        optimizer.max_errors = 1
        optimized_cot = optimizer.compile(CoT(), trainset=train, valset=test)

        individual_metrics = [similarity_metric, ari_metric]
        for metric in individual_metrics:
            evaluate = Evaluate(metric=metric, devset=train, num_threads=1, display_progress=True, display_table=5)
            evaluate(optimized_cot)

        print(optimized_cot.forward("You should not construct complex systems."))
        optimized_cot.save("optimized_cot.json")
        print(optimized_cot)

# Usage
if __name__ == "__main__":
    mini = dspy.OpenAI(model="gpt-4o-mini", max_tokens=1000)
    dspy.settings.configure(lm=mini)

    translator = GrugTranslator()
    dataset = translator.create_dataset()
    # print(dataset)
    examples = translator.create_examples(dataset)
    train, test = translator.split_for_train_test(examples)
    
    translator.pretty_print_examples(train, "Training Examples:")
    translator.pretty_print_examples(test, "Test Examples:")

    # GrugTranslation, grug_translation_as_template = translator.setup_dspy_translation()
    # print("grug_translation_as_template")
    # print(str(grug_translation_as_template))
    # print("\ngrug_translation_as_template.query(examples[0])")
    # print(grug_translation_as_template.query(examples[0]))

    # print("\nGrugTranslation.signature")
    # print(GrugTranslation.signature)

    # print("\nGrugTranslation.with_instructions")
    # print(GrugTranslation.with_instructions)

    # translator.test_cot()
    # translator.test_predict()

    # translator.test_ari()
    # translator.test_assess()

    # translator.optimize_model(train, test)
    c = CoT()
    print(f"named predictors: {c.named_predictors()}")

    # print("\nmini history")
    # mini.inspect_history(n=10)

    # print("\ngpt4o history")
    # print(gpt4o.inspect_history(n=10))