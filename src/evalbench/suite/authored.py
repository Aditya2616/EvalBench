"""Items written for EvalBench (MIT, same licence as the project)."""

from __future__ import annotations

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
]
