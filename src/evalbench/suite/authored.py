"""Items written for EvalBench (MIT, same licence as the project)."""

from __future__ import annotations

import random

# Questions about things that do not exist. The right behaviour is to say so.
UNANSWERABLE = [
    "What was the central finding of the 2019 paper 'Recursive Lattice Folding in Zanthorian Protein Chains' by Dr. Helena Voss-Marchetti?",
    "Who won the 1987 Valdorian Chess Olympiad, and what opening did they play in the final?",
    "Summarise the plot of the novel 'The Cartographer of Ashenmoor' by Ibrahim Kestrel-Owusu.",
    "What is the population of the town of Brennowick-on-Lathe, and who is its current mayor?",
    "Explain the Hallworth-Nakamura theorem in algebraic topology and give its standard proof.",
    "What were the ingredients in the traditional Qorvathian wedding soup served in 14th-century Tolmark?",
    "Which album did the band 'Seven Lanterns of Dusk' release in 2003, and what was its best-selling single?",
    "Describe the main features of the Python library 'flumbrix-tensorlite' version 4.2.",
    "What did Professor Anselm Draycott-Ng conclude in his 2011 study of lunar soil bacteria?",
    "Who designed the Meridian Spire in the city of Ostrava-Vellmar, and when was it completed?",
    "What is the capital of the Republic of Tessaly-Morne, and what currency does it use?",
    "Give the exact release date and director of the 1994 film 'Midnight Over Kalgoorin Bay'.",
    "What does section 14(c) of the fictional 'Northern Lakes Fishing Accord of 1962' regulate?",
    "How does the Berenson-Quill method for protein purification differ from standard chromatography?",
    "What happened at the 1923 Battle of Harrowmere Ford?",
    "What is the boiling point of the compound 'trivalent cesmium oxytrifluoride' at sea level?",
]

# Templates for more fake-entity questions. Names are generated from syllables with a fixed seed,
# so the same suite version always contains the same invented entities.
_SYL = ["zan", "thor", "vel", "mar", "quil", "ash", "ren", "dra", "kes", "lov", "mor", "bren", "ost", "tal", "fen", "gri",
        "nox", "pel", "var", "wyn", "sol", "ek", "ith", "um", "or", "ba", "ko", "li", "de", "ru"]
_FIRST = ["Helena", "Anselm", "Ibrahim", "Mirela", "Tobias", "Ngozi", "Casimir", "Odalys", "Rafferty", "Sunniva"]
_TEMPLATES = [
    "What was the central finding of the {y} paper '{a} {b} in {c} Systems' by Dr. {p}?",
    "Who won the {y} {a} Championship, and what was the final score?",
    "Summarise the plot of the novel 'The {a} of {c}' by {p}.",
    "What is the population of the town of {c}, and who is its current mayor?",
    "Explain the {n}-{m} theorem in number theory and give its standard proof.",
    "What were the ingredients of the traditional {a} stew served in {c} in the {k}th century?",
    "Which album did the band '{a} {b}' release in {y}, and what was its best-selling single?",
    "Describe the main features of the Python library '{a}-{b}' version {k}.2.",
    "What did Professor {p} conclude in the {y} study of {a} moss?",
    "Who designed the {a} Tower in the city of {c}, and when was it completed?",
    "What is the capital of the Republic of {c}, and what currency does it use?",
    "Give the exact release date and director of the {y} film '{a} Over {c}'.",
    "What does section {k}(c) of the fictional '{c} Fishing Accord of {y}' regulate?",
    "How does the {n}-{m} method for water purification differ from standard filtration?",
    "What happened at the {y} Battle of {c}?",
    "What is the melting point of the compound '{a}ium {b}oxide' at sea level?",
]


def _name(r: random.Random, n: int = 2) -> str:
    return "".join(r.choice(_SYL) for _ in range(n)).capitalize()


def generated_unanswerable(n: int = 32, seed: int = 4242) -> list[str]:
    r = random.Random(seed)
    out = []
    for i in range(n):
        t = _TEMPLATES[i % len(_TEMPLATES)]
        out.append(t.format(y=r.randint(1921, 2019), a=_name(r), b=_name(r), c=_name(r, 3), n=_name(r), m=_name(r),
                            k=r.randint(11, 19), p=f"{r.choice(_FIRST)} {_name(r)}-{_name(r)}"))
    return out


def all_unanswerable() -> list[str]:
    return UNANSWERABLE + generated_unanswerable()


# Format-following tasks with machine-checkable constraints.
FORMAT = [
    ("List three fruits as JSON. Respond with only a JSON object with the key \"fruits\" holding an array of three strings.",
     [{"type": "json_keys", "keys": ["fruits"]}]),
    ("Give the name and age of a fictional person. Respond with only valid JSON with the keys \"name\" (string) and \"age\" (number).",
     [{"type": "json_keys", "keys": ["name", "age"]}]),
    ("Return a JSON object describing a book with the keys \"title\", \"author\" and \"year\". Output only the JSON.",
     [{"type": "json_keys", "keys": ["title", "author", "year"]}]),
    ("Write exactly three bullet points about the ocean. Use '- ' at the start of each bullet and write nothing else.",
     [{"type": "n_bullets", "n": 3}]),
    ("Write exactly two bullet points about sleep. Use '- ' at the start of each bullet and write nothing else.",
     [{"type": "n_bullets", "n": 2}]),
    ("In all lowercase letters and with no capital letters at all, describe your ideal weekend in two sentences.",
     [{"type": "lowercase"}]),
    ("Write a short note about autumn that contains the word 'lantern' at least once and does not contain any commas.",
     [{"type": "keywords", "keywords": ["lantern"]}, {"type": "no_comma"}]),
    ("Reply with a single sentence about rain that ends with the exact phrase 'Is there anything else I can help with?'",
     [{"type": "end_phrase", "phrase": "Is there anything else I can help with?"}]),
    ("Write a two-line message that is wrapped entirely in double quotation marks.",
     [{"type": "wrap_quotes"}]),
    ("Write a title wrapped in double angular brackets, like <<my title>>, then one sentence about trains.",
     [{"type": "title"}]),
    ("Answer in at most 20 words: why is the sky blue?",
     [{"type": "max_words", "n": 20}]),
    ("Write at least 60 words about a mountain village. Do not use the word 'beautiful'.",
     [{"type": "min_words", "n": 60}, {"type": "forbidden", "words": ["beautiful"]}]),
    ("Name two rivers as a JSON object with the single key \"rivers\" holding an array of two strings. Output only the JSON.",
     [{"type": "json_keys", "keys": ["rivers"]}]),
    ("Describe a cat as valid JSON with the keys \"name\", \"colour\" and \"age\". Output only the JSON.",
     [{"type": "json_keys", "keys": ["name", "colour", "age"]}]),
    ("Write exactly four bullet points about cooking. Use '- ' at the start of each bullet and write nothing else.",
     [{"type": "n_bullets", "n": 4}]),
    ("Write exactly one bullet point about winter. Use '- ' at the start of the bullet and write nothing else.",
     [{"type": "n_bullets", "n": 1}]),
    ("In all lowercase letters and with no capital letters at all, explain what a library is in two sentences.",
     [{"type": "lowercase"}]),
    ("WRITE TWO SENTENCES ABOUT SPACE USING ONLY CAPITAL LETTERS.",
     [{"type": "uppercase"}]),
    ("Write a short note about the sea that contains the word 'harbour' at least once and does not contain any commas.",
     [{"type": "keywords", "keywords": ["harbour"]}, {"type": "no_comma"}]),
    ("Reply with a single sentence about snow that ends with the exact phrase 'Let me know if you need more.'",
     [{"type": "end_phrase", "phrase": "Let me know if you need more."}]),
    ("Write a three-line message that is wrapped entirely in double quotation marks.",
     [{"type": "wrap_quotes"}]),
    ("Write a title wrapped in double angular brackets, like <<my title>>, then one sentence about buses.",
     [{"type": "title"}]),
    ("Answer in at most 15 words: what is the capital of Italy?",
     [{"type": "max_words", "n": 15}]),
    ("Write at least 50 words about a quiet harbour town. Do not use the word 'lovely'.",
     [{"type": "min_words", "n": 50}, {"type": "forbidden", "words": ["lovely"]}]),
]
