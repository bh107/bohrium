import subprocess
import shutil
import glob
import sys
import re
import os

def create_svg(src_dir, dst_dir):
    print src_dir

    formats = ['svg', 'png', 'html']
    for s in glob.glob(src_dir+os.sep+"graph*.dot"):
        path, filename = os.path.split(s)        
        dst_path = dst_dir +os.sep+ filename
        for fmt in formats:
            end = dst_path.replace('dot', fmt)
            p = subprocess.Popen([
                'dot','-T', 'svg', s, '-o' + end
            ])
            print s, end

def main():
    create_svg(sys.argv[1], sys.argv[2])

if __name__ == "__main__":
    main()
