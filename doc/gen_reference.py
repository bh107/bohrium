#!/usr/bin/env python
import pprint
import sys
import os
from Cheetah.Template import Template

prefix  = "../"
file_types = ['.c', '.h', '.hpp', '.cpp']
paths   = [
    'core',
    'include',
    'vem',
    've'
]

template = ""

""

class Tree:

    def init(self, cargo, children):
        self.cargo      = cargo
        self.children   = childen

    def __str__(self):
        return str(self.cargo)

def node( subjects ):
    if len(subjects)>1:
        return { subjects[0]: [ node( subjects[1:] ) ] }
    else:
        return subjects[0]
    

def main():
    
    # Grab the files
    all_files   = ((root, files) for path in paths for root, dirs, files in os.walk(prefix+path) )
    filtered    = (root+os.sep+fn for root, files in all_files for fn in files if os.path.splitext(fn)[1] in file_types )
    split       = (fn.split('/') for fn in filtered)

    for subject in split:
        print node( subject )
    

if __name__ == "__main__":
    main()
