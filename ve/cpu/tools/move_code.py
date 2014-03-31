import shutil
import glob
import sys
import re
import os

def move_code(src_dir, dst_dir):
    print "Moving code from %s to %s" % (src_dir, dst_dir)
    for s in glob.glob(src_dir+"*.c"):
        path, filename = os.path.split(s)        
        m = re.match('(KRN_[0-9]+)_[0-9a-zA-Z]+.c', filename)

        src_path = path+os.sep+filename
        dst_path = dst_dir+os.sep+m.group(1)+'.c'
        shutil.move(src_path, dst_path)

def main():
    move_code(sys.argv[1], sys.argv[2])

if __name__ == "__main__":
    main()
