import os
from typing import List

import toml
from jinja2 import Environment, FileSystemLoader

_curr_dir = os.path.dirname(__file__)


def load_component_template():
    template_path = os.path.join(_curr_dir, "..", "env", "primitives", "template.toml")
    with open(template_path, "r") as f:
        template = toml.load(f)
    return template


component_template = load_component_template()


def build_context(component_names: List[str]):
    context = {
        "meta": component_template["meta"],
        "components": [],
        "model": component_template["model"],
    }

    for component_name in component_names:
        v = component_template["components"][component_name]

        context["components"].append(v)

    return context


script_env = Environment(
    loader=FileSystemLoader(_curr_dir), trim_blocks=True, lstrip_blocks=True
)


def render_script(context: dict, **extra):
    template = script_env.get_template("pipeline.py.jinja")
    return template.render(context, **extra)


notebook_env = Environment(
    loader=FileSystemLoader(_curr_dir), trim_blocks=False, lstrip_blocks=False
)


def valid_json(s: str):
    return s.replace("\n", "\\n").replace('"', '\\"')


notebook_env.filters["valid_json"] = valid_json


def render_notebook(context: dict, **extra):
    template = notebook_env.get_template("notebook.ipynb.jinja")
    return template.render(context, **extra)
