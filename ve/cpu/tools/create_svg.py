import subprocess
import shutil
import glob
import sys
import re
import os

def create_svg(src_dir, dst_dir):
    print src_dir
    for s in glob.glob(src_dir+os.sep+"graph*.dot"):
        path, filename = os.path.split(s)        
        dst_path = dst_dir +os.sep+ filename.replace('.dot', '.svg')
        p = subprocess.Popen(['dot','-T', 'svg', s, '-o'+dst_path])
        print s, dst_path


def main():
    create_svg(sys.argv[1], sys.argv[2])

if __name__ == "__main__":
    main()
