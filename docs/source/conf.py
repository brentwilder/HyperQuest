# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'hyperquest'
copyright = '2025, Brent Wilder'
author = 'Brent Wilder'

release = '0.1'
version = '0.1.12'
license = 'MIT'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output
html_theme = 'sphinx_rtd_theme'

# Add custom CSS
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]

# -- Options for EPUB output
epub_show_urls = 'footnote'

