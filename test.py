#!/usr/bin/env python
# coding=utf-8

from collections import defaultdict
import json
import re
import subprocess
import sys
from tempfile import NamedTemporaryFile
import unittest

import jsonschema
from nose.tools import *

TEST_STRING = """This is a test sentence.
This is another test sentence.
This is a third sentence with an aorsnarnt token.
This is a sentence with a special \u201c token.
"""

with open("/schemas/language_model_spec.json", "r") as spec_f:
    LANGUAGE_MODEL_SPEC_SCHEMA = json.load(spec_f)

SURPRISAL_RE = re.compile(r"sentence_id\ttoken_id\ttoken\tsurprisal\n"
                           "(\d+\s+\d+\s+[\w.<>]+\s+[-\d.]+\n)+(\d+\s+\d+\s+[\w.<>]+\s+[-\d.]+)",
                          flags=re.MULTILINE)


def get_spec():
    return json.loads(subprocess.check_output(["spec"]).decode("utf-8"))


def test_spec():
    """
    Container should return a valid specification.
    """
    try:
        jsonschema.validate(instance=get_spec(), schema=LANGUAGE_MODEL_SPEC_SCHEMA)
    except jsonschema.exceptions.ValidationError as e:
        # Avoid printing enormous schema JSONs.
        from pprint import pformat
        pinstance = pformat(e.instance, width=72)
        if len(pinstance) > 1000:
            pinstance = pinstance[:1000] + "..."

        raise ValueError("Spec validation failed for spec instance: %s" % pinstance)


class LMProcessingTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # super(LMTest, cls).setUpClass()
        cls.spec = get_spec()

        if sys.version_info[0] == 2:
            text_f = NamedTemporaryFile("w")
        else:
            text_f = NamedTemporaryFile("w", encoding="utf-8")

        with text_f:
            test_string = TEST_STRING
            if sys.version_info[0] == 2:
                test_string = TEST_STRING.encode("utf-8")
            text_f.write(test_string)
            text_f.flush()

            print("== tokenize %s" % text_f.name)
            cls.tokenized_output = subprocess.check_output(["tokenize", text_f.name]).decode("utf-8")
            print(cls.tokenized_output)
            print("\n")

            print("== unkify %s" % text_f.name)
            cls.unkified_output = subprocess.check_output(["unkify", text_f.name]).decode("utf-8")
            print(cls.unkified_output)
            print("\n")

            print("== get_surprisals %s" % text_f.name)
            cls.surprisals_output = subprocess.check_output(["get_surprisals", text_f.name]).decode("utf-8")
            print(cls.surprisals_output)

        cls.tokenized_lines = [line.strip() for line in cls.tokenized_output.strip().split("\n")]
        cls.unkified_lines = [line.strip() for line in cls.unkified_output.strip().split("\n")]
        cls.surprisal_lines = [line.strip().split("\t") for line in cls.surprisals_output.strip().split("\n")]

    @property
    def _parsed_surprisals(self):
        return self._get_parsed_surprisals(self.surprisal_lines)

    def _get_parsed_surprisals(self, surprisal_lines):
        surprisals = defaultdict(dict)
        for line in surprisal_lines[1:]:
            sentence_id, token_id, token, surprisal = line
            surprisals[int(sentence_id)][int(token_id)] = (token, float(surprisal))

        return surprisals

    def test_tokenize(self):
        for tokenized_line in self.tokenized_lines:
            for token in tokenized_line.split(" "):
                assert token in self.spec["vocabulary"]["items"], \
                        "%s missing from model vocabulary spec, but output by tokenize" % token

    def test_tokenization_match_surprisals(self):
        # token sequence output from `get_surprisals` should match token
        # sequences from `tokenize`
        surprisals = self._parsed_surprisals
        for i, tokenized_line in enumerate(self.tokenized_lines):
            tokens = tokenized_line.split(" ")
            if i == 2:
                # skip this line -- we know it has an unk
                continue

            eq_(tokens, [surprisals[i + 1][j + 1][0] for j in range(len(tokens))], "Token sequences should match exactly")

    def test_unkification(self):
        if self.spec["tokenizer"]["type"] != "word":
            return

        # same number of lines as tokenized sentences
        eq_(len(self.unkified_lines), len(self.tokenized_lines))

        # unkified sequences should match tokenized sequences in length
        for unk_line, tok_line in zip(self.unkified_lines, self.tokenized_lines):
            eq_(len(unk_line.split(" ")), len(tok_line.split(" ")))

        # dummy token should definitely be unk for any model!
        eq_(self.unkified_lines[2].split(" ")[7], "1")

    def test_surprisal_output_format(self):
        ok_(SURPRISAL_RE.match(self.surprisals_output))

    def test_surprisal_parse(self):
        eq_(set(int(line[0]) for line in self.surprisal_lines[1:]), {1, 2, 3, 4}, "Sentences retained")
        for line in self.surprisal_lines[1:]:
            # attempt to parse surprisal
            surp = float(line[3])
            ok_(surp >= 0, "valid surprisal")

    def test_surprisal_determinism(self):
        """
        Test that `get_surprisals` output is consistent across multiple calls.
        """
        # TODO
        ...


if __name__ == "__main__":
    import nose
    nose.runmodule()
