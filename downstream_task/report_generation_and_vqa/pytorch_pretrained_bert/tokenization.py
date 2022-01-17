# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Changes have been made over the original file
# https://github.com/huggingface/pytorch-transformers/blob/v0.4.0/pytorch_pretrained_bert/tokenization.py

"""Tokenization classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import unicodedata
import os
import logging

from .file_utils import cached_path
import heapq
import numpy as np

logger = logging.getLogger(__name__)

PRETRAINED_VOCAB_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-vocab.txt",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-vocab.txt",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-vocab.txt",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt",
}
PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP = {
    'bert-base-uncased': 512,
    'bert-large-uncased': 512,
    'bert-base-cased': 512,
    'bert-large-cased': 512,
    'bert-base-multilingual-uncased': 512,
    'bert-base-multilingual-cased': 512,
    'bert-base-chinese': 512,
}
VOCAB_NAME = 'vocab.txt'


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    # mapping unused tokens to special tokens
    extra_map = {}
    extra_map['[unused1]'] = '[X_SEP]'
    for i in range(10):
        extra_map['[unused{}]'.format(i+2)] = '[SEP_{}]'.format(i)

    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            if token in extra_map:
                token = extra_map[token]
            vocab[token] = index
            index += 1
    return vocab


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a peice of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class BertTokenizer(object):
    """Runs end-to-end tokenization: punctuation splitting + wordpiece"""

    def __init__(self, vocab_file, do_lower_case=True, max_len=None, never_split=("[UNK]", "[SEP]", "[X_SEP]", "[PAD]", "[CLS]", "[MASK]")):
        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(vocab_file))
        self.vocab = load_vocab(vocab_file)
        # print('self.vocab', len(self.vocab))
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])
        self.basic_tokenizer = BasicTokenizer(
            do_lower_case=do_lower_case, never_split=never_split)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
        self.max_len = max_len if max_len is not None else int(1e12)

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            if token not in self.vocab:
                # print(token)
                raise
            ids.append(self.vocab[token])
        if len(ids) > self.max_len:
            raise ValueError(
                "Token indices sequence length is longer than the specified maximum "
                " sequence length for this BERT model ({} > {}). Running this"
                " sequence through BERT will result in indexing errors".format(
                    len(ids), self.max_len)
            )
        return ids

    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens

    @classmethod
    def from_pretrained(cls, pretrained_model_name, cache_dir=None, *inputs, **kwargs):
        """
        Instantiate a PreTrainedBertModel from a pre-trained model file.
        Download and cache the pre-trained model file if needed.
        """
        if pretrained_model_name in PRETRAINED_VOCAB_ARCHIVE_MAP:
            vocab_file = PRETRAINED_VOCAB_ARCHIVE_MAP[pretrained_model_name]
        else:
            vocab_file = pretrained_model_name
        if os.path.isdir(vocab_file):
            vocab_file = os.path.join(vocab_file, VOCAB_NAME)
        # redirect to the cache, if necessary
        try:
            resolved_vocab_file = cached_path(vocab_file, cache_dir=cache_dir)
        except FileNotFoundError:
            logger.error(
                "Model name '{}' was not found in model name list ({}). "
                "We assumed '{}' was a path or url but couldn't find any file "
                "associated to this path or url.".format(
                    pretrained_model_name,
                    ', '.join(PRETRAINED_VOCAB_ARCHIVE_MAP.keys()),
                    vocab_file))
            return None
        if resolved_vocab_file == vocab_file:
            logger.info("loading vocabulary file {}".format(vocab_file))
        else:
            logger.info("loading vocabulary file {} from cache at {}".format(
                vocab_file, resolved_vocab_file))
        if pretrained_model_name in PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP:
            # if we're using a pretrained model, ensure the tokenizer wont index sequences longer
            # than the number of positional embeddings
            max_len = PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP[pretrained_model_name]
            kwargs['max_len'] = min(kwargs.get('max_len', int(1e12)), max_len)
        # Instantiate tokenizer.
        tokenizer = cls(resolved_vocab_file, *inputs, **kwargs)
        return tokenizer


class WhitespaceTokenizer(object):
    def tokenize(self, text):
        return whitespace_tokenize(text)


class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case=True, never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        """Constructs a BasicTokenizer.

        Args:
          do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case
        self.never_split = never_split

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = self._clean_text(text)
        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case and token not in self.never_split:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        if text in self.never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer`.

        Returns:
          A list of wordpiece tokens.
        """

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
                # print('bad!', text.encode('utf-8'))
            else:
                output_tokens.extend(sub_tokens)
                # print('good')
        return output_tokens


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def load_subword_nmt_table(path):
    """
    :param path: path to merge_table with subword-nmt format
    """
    table = dict()
    cur_priority = 1
    with open(path) as f:
        for line in f:
            if '#version' in line:
                continue
            token_1, token_2 = line.rstrip('\n').split(' ')
            table[(token_1, token_2)] = int(cur_priority)
            cur_priority += 1 
    return table


def load_merge_table(path):
    """
    :param path: path to merge_table
    """
    table = dict()
    with open(path) as f:
        for line in f:
            token_1, token_2, priority = line.split('\t')
            table[(token_1, token_2)] = int(priority)
            
    return table


def tokenize_word(merge_rules, word, dropout=0.0, 
                  random_generator=np.random.RandomState(), 
                  sentinels=['^', '$'],
                  regime='begin',
                  bpe_symbol='`',
                  always_merge_sentinels=True):
    """ Tokenize word using bpe merge rules
    :param merge_rules: dict [(a,b)] -> id, merge table, ids are in increasing order
    :param word: string
    :param dropout: float, dropout rate
    :param random_generator: random generator with .rand() method
    :param sentinels: list of two strings, beginning of word sentinel and end of word sentinel (empty string means that no corresponding sentinel is applied)
    :param regime:
        'begin' -- add bpe symbol to the beginning of bpe token
        'end' -- add bpe symbol to the end of bpe token
    :param bpe_symbol: str, could be one of '`', '@@', '▁'
    :param always_merge_sentinels: bool, if True, sentinels are always concatenated 
        to the first and last characters before applying BPE merges (True is equivalent to subword-nmt>=0.2, False is equivalent to subword-nmt<0.2)
    """
    
    # Subword tokens
    sw_tokens = list(word)

    # Add sentinels
    if always_merge_sentinels:
        sw_tokens = [sentinels[0] + sw_tokens[0]] + sw_tokens[1:]
        sw_tokens = sw_tokens[:-1] + [sw_tokens[-1] + sentinels[1]]
    else:
        beg_sentinel = [sentinels[0]] if len(sentinels[0]) > 0 else []
        end_sentinel = [sentinels[1]] if len(sentinels[1]) > 0 else []
        sw_tokens = beg_sentinel + sw_tokens + end_sentinel

    # Add start merges
    # Heap with pairs (priority, position)
    merge_heap = []

    for pos in range(len(sw_tokens) - 1):
        cur_nxt_pair = (sw_tokens[pos], sw_tokens[pos + 1])
        if cur_nxt_pair in merge_rules:
            cur_priority = merge_rules[cur_nxt_pair]
            merge_heap.append([cur_priority, pos])

    heapq.heapify(merge_heap)

    sw_length = len(sw_tokens)
    dropped_merges = []

    while len(merge_heap):
        cur_priority, cur_pos = heapq.heappop(merge_heap)

        # Delete not valid merges
        if cur_pos > sw_length - 2:
            continue
        cur = sw_tokens[cur_pos]
        nxt = sw_tokens[cur_pos + 1]

        if merge_rules.get((cur, nxt), None) != cur_priority:
            continue

        # Apply dropout
        if random_generator.rand() < dropout:
            dropped_merges.append([cur_priority, cur_pos])
            continue

        sw_tokens[cur_pos:cur_pos + 2] = [cur + nxt]
        sw_length -= 1

        for pair in merge_heap:
            if pair[1] > cur_pos:
                pair[1] -= 1

        # Add dropped merges back
        for priority, position in dropped_merges:
            if position > cur_pos:
                position -= 1
            heapq.heappush(merge_heap, [priority, position])

        dropped_merges = []

        # Add new possible merge
        new_cur = sw_tokens[cur_pos]
        if cur_pos > 0:
            prev = sw_tokens[cur_pos - 1]
            if (prev, new_cur) in merge_rules:
                heapq.heappush(merge_heap, [merge_rules[(prev, new_cur)], cur_pos - 1])

        if cur_pos < (sw_length - 1):
            new_next = sw_tokens[cur_pos + 1]
            if (new_cur, new_next) in merge_rules:
                heapq.heappush(merge_heap, [merge_rules[(new_cur, new_next)], cur_pos])
    
    
    sw_tokens[0] = sw_tokens[0].replace(sentinels[0], '')
    sw_tokens[-1] = sw_tokens[-1].replace(sentinels[1], '')
    
    if regime == 'begin':
        for i in range(1, sw_length):
            sw_tokens[i] = bpe_symbol + sw_tokens[i]
            
        if sw_tokens[0] == '':
            sw_tokens = sw_tokens[1:]
            sw_tokens[0] = sw_tokens[0].lstrip(bpe_symbol)
        if sw_tokens[-1] == bpe_symbol:
            sw_tokens.pop()
    elif regime == 'end':
        for i in range(sw_length -1):
            sw_tokens[i] = sw_tokens[i] + bpe_symbol
        if sw_tokens[0] == bpe_symbol:
            sw_tokens.pop(0)
        if sw_tokens[-1] == '':
            sw_tokens = sw_tokens[:-1]
            sw_tokens[-1] = sw_tokens[-1].rstrip(bpe_symbol)
        
    return sw_tokens


def tokenize_text(rules, line, dropout=0.0, random_generator=np.random.RandomState(), **args):
    return ' '.join([' '.join(tokenize_word(rules, word, dropout, random_generator, **args)) for word in line.split(' ')])


class BpeOnlineTokenizer:
    """
    Apply bpe tokenization to str line
    """
    def __init__(self, bpe_dropout_rate, merge_table, random_seed=None):
        """
        :param bpe_dropout_rate: float [0,1)
        :param merge_table: dict [(token_1, token_2)] -> priority
        """
        self.random_generator = np.random.RandomState(random_seed)
        self.bpe_dropout_rate = bpe_dropout_rate
        self.merge_table = merge_table

    def __call__(self, line, **args):
        """
        :param line: str
        :return:
        """
        return tokenize_text(self.merge_table, line, self.bpe_dropout_rate, self.random_generator, **args)
    

class BpeOnlineParallelApplier:
    """
    Apply bpe online to data in parallel
    """
    def __init__(self, bpe_dropout_rates, merge_tables, random_seed=42):
        """
        :param bpe_dropout_rate: float [0,1)
        :param merge_table: dict [(token_1, token_2)] -> priority
        """
        assert len(bpe_dropout_rates) == len(merge_tables)
        self.bpe_appliers = []
        for rate, table in zip(bpe_dropout_rates, merge_tables):
            if table is not None:
                self.bpe_appliers.append(BpeOnlineTokenizer(rate, table, random_seed))
            else:
                self.bpe_appliers.append(lambda x: x)

    def __call__(self, lines):
        assert len(self.bpe_appliers) == len(lines)
        return tuple(applier(l) for applier, l in zip(self.bpe_appliers, lines))