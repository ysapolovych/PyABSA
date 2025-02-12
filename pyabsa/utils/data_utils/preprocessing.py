# -*- coding: utf-8 -*-
# file: preprocessing.py
# time: 19:28 2023/3/1
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2021. All Rights Reserved.
import json

import pandas
import regex as re


def parse_data_dict(data_instance):
    """
    Parse text to dict
    :param data_instance: text to be parsed
    :return:

    """

    if isinstance(data_instance, dict):
        return data_instance

    elif isinstance(data_instance, str):
        try:
            return json.loads(data_instance)
        except Exception as e:
            return data_instance

    elif isinstance(data_instance, pandas.DataFrame):
        return data_instance.to_dict(orient="records")


def prep_text_for_apc_simple(
    text: str,
    start: int,
    end: int,
) -> dict[str, str]:
    """Prepares text for Aspect-based Polarity Classification by marking aspect with special tokens.
    Takes a text string and positions of aspect within text, returns marked text with special
    aspect boundary tokens and extracted aspect.
    Args:
        text (str): Input text containing the aspect to be marked
        start (int): Starting position of aspect in text
        end (int): Ending position of aspect in text
    Returns:
        dict[str, str | int]: Dictionary containing:
            - 'text': Text with aspect marked by [B-ASP] and [E-ASP] tokens
            - 'aspect': Extracted aspect substring from original text
    Example:
        >>> prep_text_for_apc_simple("The food was great", 4, 8)
        {'text': 'The [B-ASP]food[E-ASP] was great', 'aspect': 'food'}
    """
    return {
        'text': text[:m.start()] + '[B-ASP]' + m.group() + '[E-ASP]' + text[m.end():],
        'aspect': text[start:end]
    }

def prep_text_for_apc(
    text: str,
    aspect: str,
    clean_aspect: bool = False,
    allow_aspect_substring: bool = False,
    allow_fuzzy: bool = True,
) -> list[dict[str, str | int]] | None:
    """Prepare text for aspect-based sentiment analysis by marking aspect terms with special tokens.
    This function searches for aspect terms in text and wraps them with [B-ASP] and [E-ASP] tokens.
    Parameters
    ----------
    text : str
        The input text to process
    aspect : str
        The aspect term to search for in the text
    clean_aspect : bool, optional
        Whether to clean the aspect term before processing, by default False
    allow_aspect_substring : bool, optional
        If True, allows matching substrings of the aspect, by default False
    allow_fuzzy : bool, optional
        If True, allows fuzzy aspect matching in case exact matching fails, by default True    Returns
    -------
    list[dict] or None
        A list of dictionaries containing:
            - text: The processed text with aspect markers
            - aspect: The matched aspect term
            - start: Starting position of the aspect in original text
            - end: Ending position of the aspect in original text
        Returns None if no matches found
    Examples
    --------
    >>> prep_text_for_apc("The food was great", "food")
    [{
        'text': 'The [B-ASP]food[E-ASP] was great',
        'aspect': 'food',
        'start': 4,
        'end': 8
    }]
    """
    if clean_aspect:
        aspect = clean_aspect_string(aspect)

    escaped = re.escape(aspect)


    pattern: str

    if allow_aspect_substring:
        pattern = fr'({escaped}\W*)'
    else:
        pattern = fr'({escaped})(?=[,;:\.\-\)\]]|\s+|\Z)'

    matches = list(re.finditer(pattern, text))

    # If no matches, try extended pattern
    if not matches:
        if not allow_fuzzy:
            return None
        # TODO: perhaps these could be set manually?
        extended_patterns: list[str] = [fr'({escaped}(?:\'s|’s))', fr'({escaped})' + '{e<=2}']

        for exp_pat in extended_patterns:
            extended_matches = list(re.finditer(exp_pat, text))
            if extended_matches is not None:
                pattern = exp_pat
                matches = extended_matches
                break

    if not matches:
        return None

    out: list[dict] = [
        {
            "text": text[:m.start()] + '[B-ASP]' + m.group() + '[E-ASP]' + text[m.end():],
            "aspect": m.group(),
            "start": m.start(),
            "end": m.end()
        } for m in matches
    ]

    return out


def clean_aspect_string(text: str) -> str:
    """Clean and normalize aspect text by removing unbalanced brackets and quotes.
    This function processes a given aspect text by:
    1. Removing leading/trailing whitespace
    2. Removing leading/trailing punctuation (.,;)
    3. Removing unbalanced brackets/parentheses pairs
    4. Removing unmatched quotes
    Parameters
    ----------
    text : str
        The aspect text to clean
    Returns
    -------
    str
        The cleaned and normalized aspect text with unbalanced brackets and quotes removed
    Examples
    --------
    >>> clean_aspect(" (good food] ")
    "good food"
    >>> clean_aspect(",nice view.")
    "nice view"
    >>> clean_aspect("\"great service")
    "great service"
    """
    text = text.strip()

    text = re.sub(r'^[,\.;]|[,\.;]$', '', text)

    pairs: dict[str, str] = {
        '(': ')',
        '[': ']',
        '{': '}',
    }

    quotes: str = "\""

    for k, v in pairs.items():
        if text.count(k) != text.count(v):
            text = text.replace(k, '')
            text = text.replace(v, '')

    for char in quotes:
        if text.count(char) % 2 != 0:
            text = text.replace(char, '')

    return text.strip()