import util
import bohrium as bh
import bohrium.opencv

import platform
from os import environ

def has_ext():
    try:
        src    = bh.array([[1, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=bh.float32)
        kernel = bh.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bh.float32)
        bh.opencv.erode(src, kernel)
        return True
    except Exception as e:
        print("\n\033[31m[ext] Cannot test OpenCV extension methods.\033[0m")
        print(e)
        return False


class test_ext_opencv:
    def init(self):
        if not has_ext():
            return

        for t in util.TYPES.FLOAT:
            yield t

    def test_erode(self, t):
        cmd_np = "res = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=%s);" % t

        cmd_bh  = "src    = bh.array([[1, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=%s);" % t
        cmd_bh += "kernel = bh.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=%s);" % t
        cmd_bh += "res    = bh.opencv.erode(src, kernel);"
        return cmd_np, cmd_bh


    def test_dilate(self, t):
        cmd_np = "res = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=%s);" % t

        cmd_bh  = "src    = bh.array([[1, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=%s);" % t
        cmd_bh += "kernel = bh.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=%s);" % t
        cmd_bh += "res    = bh.opencv.dilate(src, kernel);"
        return cmd_np, cmd_bh


    def test_opposites(self, t):
        cmd_np = "res = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=%s);" % t

        cmd_bh  = "src    = bh.array([[1, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=%s);" % t
        cmd_bh += "kernel = bh.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=%s);" % t
        cmd_bh += "res    = bh.opencv.dilate(bh.opencv.erode(src, kernel), kernel);"
        return cmd_np, cmd_bh
