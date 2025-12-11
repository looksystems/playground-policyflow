"""Jinja2 template loader for prompt management."""

from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape

# Template directory
TEMPLATE_DIR = Path(__file__).parent

# Create Jinja2 environment
_env = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    autoescape=select_autoescape(default=False),
    trim_blocks=True,
    lstrip_blocks=True,
)


def render(template_name: str, **kwargs) -> str:
    """
    Render a Jinja2 template with the given context.

    Args:
        template_name: Name of the template file (e.g., "policy_parser.j2")
        **kwargs: Variables to pass to the template

    Returns:
        Rendered template string
    """
    template = _env.get_template(template_name)
    return template.render(**kwargs)


def get_template(template_name: str):
    """
    Get a Jinja2 template object for manual rendering.

    Args:
        template_name: Name of the template file

    Returns:
        Jinja2 Template object
    """
    return _env.get_template(template_name)
