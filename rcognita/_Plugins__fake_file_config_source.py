# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
from typing import List, Optional

from omegaconf import OmegaConf

from hydra.core.object_type import ObjectType
from hydra.plugins.config_source import ConfigLoadError, ConfigResult, ConfigSource
from io import StringIO
import re




def sub_map(pattern, f, s):
    def map_match(match):
        return f(match.group())
    return re.sub(pattern, map_match, s)

def wrap_equals_expression(content):
    i = content.index('=')
    return content[:i] + f"${{get:{content[i + 1:].lstrip()}}}"

def equals_sugar_for_inlines(content):
    return sub_map(r"(:|-)\s*\=.*\S+.*", wrap_equals_expression, content)

def wrap_tilde_expression(content):
    i = content.index('~')
    return content[:i] + f"${{same:{content[i + 1:].lstrip()}}}"

def tilde_sugar_for_references(content):
    return sub_map(r"(:|-)\s*\~.*\S+.*", wrap_tilde_expression, content)

def wrap_tilde_expression_specific(content):
    i = content.index('~')
    try:
        j = content.index('-')
        if j > i:
            raise ValueError()
    except ValueError:
        j = content.index(':')
    name = content[j + 1:i].strip()
    return content[:i] + f"${{same:{content[i + 1:].lstrip()};{name}}"

def tilde_sugar_for_specific_references(content):
    return sub_map(r"(:|-)\s*[A-Za-z0-9_]+\s*\~.*\S+.*", wrap_tilde_expression_specific, content)

def wrap_dollar_expression(content):
    i = content.index('$')
    if content[i + 1] == "{":
        return content
    return content[:i] + f"${{{content[i + 1:].lstrip()}}}"

def dolar_sugar_for_references(content):
    return sub_map(r"(:|-)\s*\$.*\S+.*", wrap_dollar_expression, content)

def fix_characters(content):
    return content.replace('(', '\\(').replace(')', '\\)').replace('[', '\\[').replace(']', '\\]').replace(',', '\\,').replace(';', ',')

def numerize_string(s):
    new_s = ""
    string_kind=""
    chr_terms = []
    for char in s:
        if string_kind:
            if char == string_kind:
                string_kind = ""
                new_s = new_s + "+".join(chr_terms) + ")"
                chr_terms = []
            else:
                code = ord(char)
                chr_terms.append(f'chr({code})')
        else:
            if char == '"' or char == "'":
                string_kind = char
                new_s = new_s + "("
            else:
                new_s = new_s + char
    return new_s


def numerize_strings_inside_braces(content):
    return sub_map(r"\{.+\}", numerize_string, content)

def pre_parse(content):
    content = dolar_sugar_for_references(content)
    content = equals_sugar_for_inlines(content)
    content = tilde_sugar_for_references(content)
    content = tilde_sugar_for_specific_references(content)
    content = numerize_strings_inside_braces(content)
    content = fix_characters(content)

    return content

class FileConfigSource(ConfigSource):
    def __init__(self, provider: str, path: str) -> None:
        if path.find("://") == -1:
            path = f"{self.scheme()}://{path}"
        super().__init__(provider=provider, path=path)

    @staticmethod
    def scheme() -> str:
        return "file"

    def load_config(self, config_path: str) -> ConfigResult:
        normalized_config_path = self._normalize_file_name(config_path)
        full_path = os.path.realpath(os.path.join(self.path, normalized_config_path))
        if not os.path.exists(full_path):
            raise ConfigLoadError(f"Config not found : {full_path}")
        with open(full_path, encoding="utf-8") as f:
            content = pre_parse(f.read())
        with StringIO(content) as f:
            header_text = f.read(512)
            header = ConfigSource._get_header_dict(header_text)
            f.seek(0)
            cfg = OmegaConf.load(f)
            return ConfigResult(
                config=cfg,
                path=f"{self.scheme()}://{self.path}",
                provider=self.provider,
                header=header,
            )

    def available(self) -> bool:
        return self.is_group("")

    def is_group(self, config_path: str) -> bool:
        full_path = os.path.realpath(os.path.join(self.path, config_path))
        return os.path.isdir(full_path)

    def is_config(self, config_path: str) -> bool:
        config_path = self._normalize_file_name(config_path)
        full_path = os.path.realpath(os.path.join(self.path, config_path))
        return os.path.isfile(full_path)

    def list(self, config_path: str, results_filter: Optional[ObjectType]) -> List[str]:
        files: List[str] = []
        full_path = os.path.realpath(os.path.join(self.path, config_path))
        for file in os.listdir(full_path):
            file_path = os.path.join(config_path, file)
            self._list_add_result(
                files=files,
                file_path=file_path,
                file_name=file,
                results_filter=results_filter,
            )

        return sorted(list(set(files)))
