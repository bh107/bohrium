Code Generator
==============

The code-templates are quite messy.

Embedding Code-Templates
========================

To avoid loading code-templates from disk, they are instead transformed from ''.tpl'' to ''.c''.
In their ''.c'' form they are stored as ''const char*'' with an associated ''size_t'' for the length of the code-template.

