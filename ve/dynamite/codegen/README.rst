Code Generator
==============

The string templates are quite messy.

Embedding snippets
==================

To avoid loading snippets from disk, they are instead transformed from ''.tpl'' to ''.c''.
In their ''.c'' form they are stored as ''const char*'' with an associated ''size_t'' for the length of the snippet.



