import clr
import System

# Support for debugging
fullpath = System.IO.Path.GetFullPath(System.IO.Directory.GetCurrentDirectory())
if not System.IO.File.Exists(System.IO.Path.Combine(fullpath, "NumCIL.dll")):
    import sys
    fullpath = System.IO.Path.GetFullPath(System.IO.Path.Combine(System.IO.Directory.GetCurrentDirectory(),  "..\\..\\..\\bridge\\NumCIL\\NumCIL\\bin\\Release"))
    if not System.IO.File.Exists(System.IO.Path.Combine(fullpath, "NumCIL.dll")):
        System.IO.Path.GetFullPath(System.IO.Path.Combine(System.IO.Directory.GetCurrentDirectory(),  "..\\..\\..\\bridge\\NumCIL\\NumCIL\\bin\\Debug"))
        if not System.IO.File.Exists(System.IO.Path.Combine(fullpath, "NumCIL.dll")):
            raise Exception("Unable to locate NumCIL.dll")

    #We hack on the path, because the IronPython builder does not copy the reference as it should
    sys.path.insert(0, fullpath)

try:
    clr.AddReference("NumCIL")
except Exception:
    raise Exception("The NumCIL.dll binary was found but could not be loaded.\n" + 
                    "This can be caused by attempting to run a script from an untrusted location, such as a network folder.\n" + 
                    "If you are attempting to run from a network folder, you need to add a config file to the IronPython interpreter (ipy.exe or ipy64.exe):\n\n" +
                    "<?xml version=\"1.0\"?>\n" +
                    "<configuration>\n" + 
                    "  <runtime>\n" + 
                    "    <loadFromRemoteSources enabled=\"true\" />\n" +
                    "  </runtime>\n" +
                    "</configuration>\n")
import NumCIL

pi = System.Math.PI

float32 = System.Single
float64 = System.Double
double = System.Double
single = System.Single

newaxis = NumCIL.Range.NewAxis

def GetNdClass(dtype):
    if dtype==System.Single:
        return NumCIL.Float
    elif dtype==System.Double or dtype==float:
        return NumCIL.Double
    elif isinstance(dtype, NumCIL.Float.NdArray):
        return NumCIL.Float
    elif isinstance(dtype, NumCIL.Double.NdArray):
        return NumCIL.Double
    elif isinstance(dtype, ndarray):
        return dtype.cls
    else:
        raise Exception("There is only support for float and double types")

def ReshapeShape(sh):
    if (isinstance(sh, list)):
        return System.Array[System.Int64](sh)
    else:
        return sh

def SliceToRange(sl):
    start = sl.start
    stop = sl.stop
    step = sl.step

    if (start == 0 or start == None) and (stop == System.Int32.MaxValue or stop == None) and step == None:
        return NumCIL.Range.All
    elif stop == System.Int32.MaxValue:
        stop = 0

    if start == None:
        start = 0
    if stop == None:
        stop = 0

    if step == None:
        return NumCIL.Range(start, stop)
    else:
        return NumCIL.Range(start, stop, step)


def SlicesToRanges(sl):
    ranges = System.Array.CreateInstance(NumCIL.Range, len(sl))
    for i in range(0, len(sl)):
        if isinstance(sl[i], slice):
            ranges[i] = SliceToRange(sl[i])
        elif isinstance(sl[i], NumCIL.Range):
            ranges[i] = sl[i]
        elif isinstance(sl[i], int) or isinstance(sl[i], System.Int64)  or isinstance(sl[i], System.Int32):
            ranges[i] = NumCIL.Range.El(sl[i])
        else:
            raise Exception("Invalid range slice " + str(type(sl[i])))

    return ranges

class ndarray:
    parent = None
    dtype = None
    cls = None

    def __init__(self, p):
        if isinstance(p, NumCIL.Float.NdArray):
            self.dtype = float32
            self.cls = NumCIL.Float
            self.parent = p
        elif isinstance(p, NumCIL.Double.NdArray):
            self.dtype = float64
            self.cls = NumCIL.Double
            self.parent = p
        elif isinstance(p, NumCIL.Generic.NdArray[float32]):
            self.dtype = float32
            self.cls = NumCIL.Float
            self.parent = NumCIL.Float.NdArray(p)
        elif isinstance(p, NumCIL.Generic.NdArray[float64]):
            self.dtype = float64
            self.cls = NumCIL.Double
            self.parent = NumCIL.Double.NdArray(p)
        else:
            raise Exception("Not yet implemented " + str(type(p)))

    def sum(self):
        return self.parent.Sum()

    def max(self):
        return self.parent.Max()

    def min(self):
        return self.parent.Min()

    def repeat(self, repeats, axis = None):
        return ndarray(self.parent.Repeat(repeats, axis))

    def getsize(self):
        return self.parent.Shape.Elements
    
    size = property(fget=getsize)

    def reshape(self, t):
        if isinstance(t, tuple):
            return ndarray(self.parent.Reshape(NumCIL.Shape(System.Array[System.Int64](list(t)))))
        else:
            return ndarray(self.parent.Reshape(NumCIL.Shape(t)))

    def getShapeTuple(self):
        return tuple([x.Length for x in self.parent.Shape.Dimensions])

    def setShapeTuple(self, t):
        self.parent.Reshape(System.Array[System.Int64](list(t)))

    shape = property(fget = getShapeTuple, fset = setShapeTuple)

    def transpose(self):
        if self.parent.Shape.Dimensions.LongLength < 2:
            return self
        else:
            return ndarray(self.parent.Transpose())

    T = property(fget=transpose)

    def __len__(self):
        return self.parent.Shape.Dimensions[0].Length

    def __getslice__(self, start, end):
        sl = slice(start, end)
        return self.__getitem__(sl)

    def __getitem__(self, slices):
        if isinstance(slices, list) or isinstance(slices, tuple):
            return ndarray(self.parent[SlicesToRanges(slices)])
        elif isinstance(slices, slice):
            return ndarray(self.parent[System.Array[NumCIL.Range]([SliceToRange(slices)])])
        elif isinstance(slices, int) or isinstance(slices, System.Int64) or isinstance(slices, System.Int32):
            return self.parent.Value[slices]
        else:
            return ndarray(self.parent[slices])

    def __setitem__(self, slices, value):
        v = value
        if isinstance(v, ndarray):
            v = v.parent
        elif (isinstance(v, float) or isinstance(v, int)) and self.cls == NumCIL.Float:
            v = System.Single(v)
        elif isinstance(v, int) and self.cls == NumCIL.Double:
            v = System.Double(v)
        elif isinstance(v, list) or isinstance(v, tuple):
            lst = System.Collections.Generic.List[self.dtype]()
            for a in v:
                if self.cls == NumCIL.Float:
                    lst.Add(System.Single(a))
                elif self.cls == NumCIL.Double:
                    lst.Add(System.Double(a))
                else:
                    raise Exception("Self cls not supported? " + str(type(self.cls)))
            c = getattr(self.cls, "NdArray")
            v = c(lst.ToArray())

        if isinstance(slices, list) or isinstance(slices, tuple):
            self.parent[SlicesToRanges(slices)] =  v
        elif isinstance(slices, slice):
            self.parent[System.Array[NumCIL.Range]([SliceToRange(slices)])] = v
        elif isinstance(slices, int):
            self.parent.Values[slices] = v
        else:
            self.parent[slices] = v


    def __add__(self, other):
        if isinstance(other, ndarray):
            return ndarray(self.cls.Add.Apply(self.parent, other.parent))
        elif type(other) == float and self.cls == NumCIL.Float:
            return ndarray(self.cls.Add.Apply(self.parent, System.Single(other)))
        else:
            return ndarray(self.cls.Add.Apply(self.parent, other))

    def __radd__(self, other):
        if isinstance(other, ndarray):
            return ndarray(self.cls.Add.Apply(other.parent, self.parent))
        elif type(other) == float and self.cls == NumCIL.Float:
            return ndarray(self.cls.Add.Apply(System.Single(other), self.parent))
        else:
            return ndarray(self.cls.Add.Apply(other, self.parent))

    def __iadd__(self, other):
        if isinstance(other, ndarray):
            self.cls.Add.Apply(self.parent, other.parent, self.parent)
            return self
        elif type(other) == float and self.cls == NumCIL.Float:
            self.cls.Add.Apply(self.parent, System.Single(other), self.parent)
            return self
        else:
            self.cls.Add.Apply(self.parent, other, self.parent)
            return self

    def __sub__(self, other):
        if isinstance(other, ndarray):
            return ndarray(self.cls.Sub.Apply(self.parent, other.parent))
        elif type(other) == float and self.cls == NumCIL.Float:
            return ndarray(self.cls.Sub.Apply(self.parent, System.Single(other)))
        else:
            return ndarray(self.cls.Sub.Apply(self.parent, other))

    def __rsub__(self, other):
        if isinstance(other, ndarray):
            return ndarray(self.cls.Sub.Apply(other.parent, self.parent))
        elif type(other) == float and self.cls == NumCIL.Float:
            return ndarray(self.cls.Sub.Apply(System.Single(other), self.parent))
        else:
            return ndarray(self.cls.Sub.Apply(other, self.parent))

    def __isub__(self, other):
        if isinstance(other, ndarray):
            self.cls.Sub.Apply(self.parent, other.parent, self.parent)
            return self
        elif type(other) == float and self.cls == NumCIL.Float:
            self.cls.Sub.Apply(self.parent, System.Single(other), self.parent)
            return self
        else:
            self.cls.Sub.Apply(self.parent, other, self.parent)
            return self

    def __div__(self, other):
        if isinstance(other, ndarray):
            return ndarray(self.cls.Div.Apply(self.parent, other.parent))
        elif type(other) == float and self.cls == NumCIL.Float:
            return ndarray(self.cls.Div.Apply(self.parent, System.Single(other)))
        else:
            return ndarray(self.cls.Div.Apply(self.parent, other))

    def __rdiv__(self, other):
        if isinstance(other, ndarray):
            return ndarray(self.cls.Div.Apply(other.parent, self.parent))
        elif type(other) == float and self.cls == NumCIL.Float:
            return ndarray(self.cls.Div.Apply(System.Single(other), self.parent))
        else:
            return ndarray(self.cls.Div.Apply(other, self.parent))

    def __idiv__(self, other):
        if isinstance(other, ndarray):
            self.cls.Div.Apply(self.parent, other.parent, self.parent)
            return self
        elif type(other) == float and self.cls == NumCIL.Float:
            self.cls.Div.Apply(self.parent, System.Single(other), self.parent)
            return self
        else:
            self.cls.Div.Apply(self.parent, other, self.parent)
            return self

    def __mul__(self, other):
        if isinstance(other, ndarray):
            return ndarray(self.cls.Mul.Apply(self.parent, other.parent))
        elif type(other) == float and self.cls == NumCIL.Float:
            return ndarray(self.cls.Mul.Apply(self.parent, System.Single(other)))
        else:
            return ndarray(self.cls.Mul.Apply(self.parent, other))

    def __rmul__(self, other):
        if isinstance(other, ndarray):
            return ndarray(self.cls.Mul.Apply(other.parent, self.parent))
        elif type(other) == float and self.cls == NumCIL.Float:
            return ndarray(self.cls.Mul.Apply(System.Single(other), self.parent))
        else:
            return ndarray(self.cls.Mul.Apply(other, self.parent))

    def __imul__(self, other):
        if isinstance(other, ndarray):
            self.cls.Mul.Apply(self.parent, other.parent, self.parent)
            return self
        elif type(other) == float and self.cls == NumCIL.Float:
            self.cls.Mul.Apply(self.parent, System.Single(other), self.parent)
            return self
        else:
            self.cls.Mul.Apply(self.parent, other, self.parent)
            return self

    def __mod__(self, other):
        if isinstance(other, ndarray):
            return ndarray(self.cls.Mod.Apply(self.parent, other.parent))
        elif type(other) == float and self.cls == NumCIL.Float:
            return ndarray(self.cls.Mod.Apply(self.parent, System.Single(other)))
        else:
            return ndarray(self.cls.Mod.Apply(self.parent, other))

    def __rmod__(self, other):
        if isinstance(other, ndarray):
            return ndarray(self.cls.Mod.Apply(other.parent, self.parent))
        elif type(other) == float and self.cls == NumCIL.Float:
            return ndarray(self.cls.Mod.Apply(System.Single(other), self.parent))
        else:
            return ndarray(self.cls.Mod.Apply(other, self.parent))

    def __imod__(self, other):
        if isinstance(other, ndarray):
            self.cls.Mod.Apply(self.parent, other.parent, self.parent)
            return self
        elif type(other) == float and self.cls == NumCIL.Float:
            self.cls.Mod.Apply(self.parent, System.Single(other), self.parent)
            return self
        else:
            self.cls.Mod.Apply(self.parent, other, self.parent)
            return self

    def __pow__(self, other, modulo = None):
        if modulo != None:
            raise Exception("Modulo version of Pow not supported")

        if isinstance(other, ndarray):
            return ndarray(self.cls.Pow.Apply(self.parent, other.parent))
        elif type(other) == float and self.cls == NumCIL.Float:
            return ndarray(self.cls.Pow.Apply(self.parent, System.Single(other)))
        else:
            return ndarray(self.cls.Pow.Apply(self.parent, other))

    def __rpow__(self, other):
        if isinstance(other, ndarray):
            return ndarray(self.cls.Pow.Apply(other.parent, self.parent))
        elif type(other) == float and self.cls == NumCIL.Float:
            return ndarray(self.cls.Pow.Apply(System.Single(other), self.parent))
        else:
            return ndarray(self.cls.Pow.Apply(other, self.parent))

    def __ipow__(self, other, modulo = None):
        if modulo != None:
            raise Exception("Modulo version of Pow not supported")
        if isinstance(other, ndarray):
            self.cls.Pow.Apply(self.parent, other.parent, self.parent)
            return self
        elif type(other) == float and self.cls == NumCIL.Float:
            self.cls.Pow.Apply(self.parent, System.Single(other), self.parent)
            return self
        else:
            self.cls.Pow.Apply(self.parent, other, self.parent)
            return self

    def __neg__ (self):
        return self.parent.Negate()

    def __abs__ (self):
        return self.parent.Abs()

    def __str__(self):
        return self.parent.ToString()

def empty(shape, dtype=float, order='C', dist=False):
    return ndarray(GetNdClass(dtype).Generate.Empty(ReshapeShape(shape)))

def ones(shape, dtype=float, order='C', dist=False):
    return ndarray(GetNdClass(dtype).Generate.Ones(ReshapeShape(shape)))

def zeroes(shape, dtype=float, order='C', dist=False):
    return ndarray(GetNdClass(dtype).Generate.Zeroes(ReshapeShape(shape)))

def zeros(shape, dtype=float, order='C', dist=False):
    return zeroes(shape, dtype, order, dist)

def arange(shape, dtype=float, order='C', dist=False):
    return ndarray(GetNdClass(dtype).Generate.Arange(ReshapeShape(shape)))

class ufunc:
    op = None
    nin = 2
    nout = 1
    nargs = 3
    name = None

    def __init__(self, op, name):
        self.op = op
        self.name = name

    def aggregate(self, a):
        if not isinstance(a, ndarray):
            raise Exception("Can only aggregate ndarrays")

        cls = a.cls
        f = getattr(cls, self.op)
        return f.Aggregate(a.parent)


    def reduce(self, a, axis=0, dtype=None, out=None, skipna=False, keepdims=False):
        if dtype != None or skipna != False or keepdims != False:
            raise Exception("Arguments dtype, skipna or keepdims are not supported")
        if not isinstance(a, ndarray):
            raise Exception("Can only reduce ndarrays")

        cls = None
        if out != None and isinstance(out, ndarray):
            cls = out.cls
        elif isinstance(a, ndarray):
            cls = a.cls

        if out != None and isinstance(out, ndarray):
            out = out.parent

        f = getattr(cls, self.op)
        return ndarray(f.Reduce(a.parent, axis, out))

    def __call__(self, a, b, out = None):
        cls = None
        if out != None and isinstance(out, ndarray):
            cls = out.cls
        elif isinstance(a, ndarray):
            cls = a.cls
        elif isinstance(b, ndarray):
            cls = b.cls

        if cls == None:
            raise Exception("Apply not supported for scalars")
        else:
            f = getattr(cls, self.op)
            if isinstance(a, ndarray):
                a = a.parent
            if isinstance(b, ndarray):
                b = b.parent
            if out != None and isinstance(out, ndarray):
                out = out.parent

            return ndarray(f.Apply(a, b, out))

add = ufunc("Add", "add")
subtract = ufunc("Sub", "subtract")
multiply = ufunc("Mul", "multiply")
divide = ufunc("Div", "divide")
mod = ufunc("Mod", "mod")
maximum = ufunc("Max", "maximum")
minimum = ufunc("Min", "minimum")

def size(x):
    if isinstance(x, ndarray):
        return x.getsize()
    else:
        raise Exception("Can only return size of ndarray")

class random:
    @staticmethod
    def random(shape, dtype=float, order='C', cphvb=False):
        return ndarray(GetNdClass(dtype).Generate.Random(ReshapeShape(shape)))

def activate_cphVB(active = True):
    try:
        clr.AddReference("NumCIL.cphVB")
    except Exception:
        raise Exception("Unable to activate NumCIL.cphVB extensions, make sure that the NumCIL.cphVB.dll is placed in the same folder as NumCIL.dll")
    
    import NumCIL.cphVB

    if active:
        NumCIL.cphVB.Utility.Activate()
    else:
        NumCIL.cphVB.Utility.Deactivate()
