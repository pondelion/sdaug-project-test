from dataclasses import dataclass, field
import itertools
import random
from typing import List


@dataclass
class ClassificationRulebaseAugPromptGenerator:
    subject_list: List[str]  # e.g. ['dog']
    adjective_list: List[str] = field(default_factory=list)  # e.g. ['cute', 'angry']
    verb_list: List[str] = field(default_factory=list)  # e.g. ['running', 'sitting']

    def gen_random_sentence(self, n: int = 1) -> List[str]:
        combinations = list(itertools.product(
            self.subject_list,
            self.adjective_list if len(self.adjective_list) > 0 else [None],
            self.verb_list if len(self.verb_list) > 0 else [None],
        ))
        random_conbinations = random.choices(combinations, k=n)
        random_sentence_list = []
        for sub, adj, verb in random_conbinations:
            sentence = sub
            if adj is not None:
                sentence = f'{adj} {sentence}'
            if verb is not None:
                sentence = f'{sentence} {verb}'
            random_sentence_list.append(sentence)
        return random_sentence_list


LLM_PROMPT_GENERATION_PROMPT_FMT_LIST = [
    f'I want to generate diverse {subject} images by generative AI model, please generate random {n_generate} prompts. Return only list in the format prompt1\nprompt2...'
]


class ClassificationLLMAugPromptGenerator:

    def __init__(
        self,
        prompt_gen_prompt_fmt_list: List[str] = LLM_PROMPT_GENERATION_PROMPT_FMT_LIST,
    ):
        self._prompt_gen_prompt_fmt_list = prompt_gen_prompt_fmt_list

    def get_prompt_generation_prompt(self, subject: str, n: int = 10) -> str:
        prompt_gen_prompt_fmt = random.sample(self._prompt_gen_prompt_fmt_list, k=1)[0]
        prompt_gen_prompt = prompt_gen_prompt_fmt.format(subject=subject, n_generate=n)
        return prompt_gen_prompt
