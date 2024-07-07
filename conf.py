# -*- coding: utf-8 -*-
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os


# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.doctest',
              'sphinx.ext.intersphinx',
              'sphinx.ext.mathjax',
              'sphinx.ext.viewcode',
              'sphinx.ext.githubpages',
              'sphinx_gallery.gen_gallery',
              'jupyterlite_sphinx',
              ]

try:
    import sphinxext.opengraph
    extensions.append('sphinxext.opengraph')
except ImportError:
    print("ERROR: sphinxext.opengraph import failed")

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'Dirty data science'
author = u'Gaël Varoquaux'
copyright = u'2021, ' + author

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
version = '2021.1'
# The full version, including alpha/beta/rc tags.
release = version

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'
# Doc: https://alabaster.readthedocs.io/en/latest/customization.html

html_sidebars = {
    '**': [
        'about.html',
        #'globallinks.html',
        'localtoc.html',
        'relations.html',
        #'searchbox.html',
    ],
    'index': [
        'about.html',
        'localtoc.html',
        'relations.html',
        #'searchbox.html',
    ]
}

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    'logo': 'piggy.svg',
    'github_user': 'dirty-data-science',
    'github_repo': 'python',
    'github_button': 'true',
    'github_type': 'star',
    'github_count': 'true',
    'show_powered_by': 'false',
    'logo_name': 'true',
    'gray_1': "#030",
    'gray_2': "#F1FFF1",
    'link': "#076B00",
#    'gray_3': "#090",
    'fixed_sidebar': 'true',
    'note_bg': "rgb(246, 248, 250);",
    #'topic_bg': "rgb(246, 248, 250);",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# Modify the title, so as to get good social-media links
html_title = "&mdash; Dirty data science"


# Configuration for intersphinx
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
    'matplotlib': ('https://matplotlib.org/', None),
    'sklearn': ('https://scikit-learn.org/stable', None),
    'skimage': ('http://scikit-image.org/docs/stable/', None),
    'mayavi': ('http://docs.enthought.com/mayavi/mayavi/', None),
    'statsmodels': ('http://www.statsmodels.org/stable/', None),
    'pandas': ('http://pandas.pydata.org/pandas-docs/stable/', None),
    'seaborn': ('http://seaborn.pydata.org/', None),
    'skrub': ('https://skrub-data.org/stable/', None),
}


# -- sphinx-gallery configuration -----------------------------------------
from sphinx_gallery.sorting import FileNameSortKey
sphinx_gallery_conf = {
    'filename_pattern': '',
    'backreferences_dir': os.path.join('generated'),
    'reference_url': {
#        'dirty_cat': 'https://dirty-cat.github.io/stable/',
        'numpy': 'http://docs.scipy.org/doc/numpy',
#        'scipy': 'http://docs.scipy.org/doc/scipy/reference',
#        'pandas': 'http://pandas.pydata.org/pandas-docs/stable',
#        'seaborn': 'http://seaborn.pydata.org/',
        'matplotlib': 'http://matplotlib.org/stable',
        'sklearn': 'http://scikit-learn.org/stable',
#        #'scikit-image': 'http://scikit-image.org/docs/stable/',
#        #'mayavi': 'http://docs.enthought.com/mayavi/mayavi/',
        #'statsmodels': 'http://www.statsmodels.org/stable/',
        },
    'examples_dirs':'notes',
    'gallery_dirs':'gen_notes',
    'within_subsection_order': FileNameSortKey,
    'download_all_examples': False,
    'binder': {
        'org': 'dirty-data-science',
        'repo': 'python',
        'binderhub_url': 'https://mybinder.org',
        'branch': 'gh-pages',
        'dependencies': ['requirements.txt',],
        'notebooks_dir': 'notes'
    },
    'jupyterlite': {
        'use_jupyter_lab': False,
    },
    "inspect_global_variables": False,
}

# -- sphinxext.opengraph configuration -------------------------------------
ogp_site_url = "https://dirtydata.science/python"
ogp_image = "https://dirtydata.science/python/_static/piggy.svg"
ogp_use_first_image = True
ogp_site_name = "Dirty Data Science"


# -- The javascript to highlight the toc as we scroll ----------------------
html_js_files = ['scrolltoc.js']
