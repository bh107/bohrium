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
int16 = System.Int16
uint16 = System.UInt16
int32 = System.Int32
uint32 = System.UInt32
int64 = System.Int64
uint64 = System.UInt64
complex64 = NumCIL.Complex64.DataType
bool = System.Boolean

clr.AddReference("System.Numerics")
complex128 = System.Numerics.Complex

newaxis = NumCIL.Range.NewAxis

def GetNdClass(dtype):
    if dtype == float32 or isinstance(dtype, NumCIL.Float.NdArray):
        return NumCIL.Float
    elif dtype == float64 or dtype == float or isinstance(dtype, NumCIL.Double.NdArray):
        return NumCIL.Double
    elif dtype == int16 or isinstance(dtype, NumCIL.Int16.NdArray):
        return NumCIL.Int16
    elif dtype == uint16 or isinstance(dtype, NumCIL.UInt16.NdArray):
        return NumCIL.UInt16
    elif dtype == int32 or isinstance(dtype, NumCIL.Int32.NdArray):
        return NumCIL.Int32
    elif dtype == uint32 or isinstance(dtype, NumCIL.UInt32.NdArray):
        return NumCIL.UInt32
    elif dtype == int64 or isinstance(dtype, NumCIL.Int64.NdArray):
        return NumCIL.Int64
    elif dtype == uint64 or isinstance(dtype, NumCIL.UInt64.NdArray):
        return NumCIL.UInt64
    elif dtype == complex64 or isinstance(dtype, NumCIL.Complex64.NdArray):
        return NumCIL.Complex64
    elif dtype == complex128 or isinstance(dtype, NumCIL.Complex128.NdArray):
        return NumCIL.Complex128
    elif dtype == bool or isinstance(dtype, NumCIL.Boolean.NdArray):
        return NumCIL.Boolean
    elif isinstance(dtype, ndarray):
        return dtype.cls
    else:
        raise Exception("The specified type is not supported: " + str(type(dtype)))

def ReshapeShape(sh):
    if (isinstance(sh, list)):
        return System.Array[System.Int64](sh)
    else:
        return sh

def SliceToRange(sl):
    if isinstance(sl, int) or isinstance(sl, long) or isinstance(sl, float) or isinstance(sl, System.Int64) or isinstance(sl, System.Int32):
        return NumCIL.Range.El(System.Int64(sl))

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
        elif isinstance(sl[i], int) or isinstance(sl[i], long) or isinstance(sl[i], float) or isinstance(sl[i], System.Int64) or isinstance(sl[i], System.Int32):
            ranges[i] = NumCIL.Range.El(System.Int64(sl[i]))
        else:
            raise Exception("Invalid range slice " + str(type(sl[i])))

    return ranges

def CreateNdArrayFromList(lst):
    dims = []
    fullsize = 0
    l = lst
    dtype = None

    # We inspect the data object to discover the shape
    # and data type
    while(isinstance(l, list) or isinstance(l, tuple)):
        dims.append(len(l))
        if fullsize == 0:
            fullsize = len(l)
        else:
            fullsize *= len(l)

        for x in l:
            if isinstance(x, list) or isinstance(x, tuple):
                if len(x) != len(l[0]):
                    raise Exception("Bad shapes")
            else:
                if isinstance(x, int):
                    if dtype == None:
                        dtype = int32
                elif isinstance(x, long):
                    if dtype == None or dtype == int32:
                        dtype = int64
                elif isinstance(x, float):
                    dtype = float64
                else:
                    raise Exception("Bad type in list: " + str(type(x)))

        
        l = l[0]

    if dtype == None:
        dtype = float64

    # Then we create a new contigous array
    arr = System.Array.CreateInstance(dtype, fullsize)
    ix = 0

    tmplst = [x for x in lst]

    while len(tmplst) > 0:
        top = tmplst[0]
        tmplst.remove(top)
        if isinstance(top, list) or isinstance(top, tuple):
            for x in top:
                tmplst.append(x)
        else:
            arr[ix] = dtype(top)
            ix += 1

    shp = NumCIL.Shape(System.Array[System.Int64]([System.Int64(x) for x in dims]))
    return GetNdClass(dtype).NdArray(arr).Reshape(shp), dtype        


class ndarray:
    parent = None
    dtype = None
    cls = None
    owncls = None
    collapsedSlicing = True

    def __init__(self, p):
        if isinstance(p, NumCIL.Float.NdArray):
            self.dtype = float32
            self.cls = NumCIL.Float
            self.parent = p
        elif isinstance(p, NumCIL.Double.NdArray):
            self.dtype = float64
            self.cls = NumCIL.Double
            self.parent = p
        elif isinstance(p, NumCIL.Int16.NdArray):
            self.dtype = int16
            self.cls = NumCIL.Int16
            self.parent = p
        elif isinstance(p, NumCIL.UInt16.NdArray):
            self.dtype = uint16
            self.cls = NumCIL.UInt16
            self.parent = p
        elif isinstance(p, NumCIL.Int32.NdArray):
            self.dtype = int32
            self.cls = NumCIL.Int32
            self.parent = p
        elif isinstance(p, NumCIL.UInt32.NdArray):
            self.dtype = uint32
            self.cls = NumCIL.UInt32
            self.parent = p
        elif isinstance(p, NumCIL.Int64.NdArray):
            self.dtype = int64
            self.cls = NumCIL.Int64
            self.parent = p
        elif isinstance(p, NumCIL.UInt64.NdArray):
            self.dtype = uint64
            self.cls = NumCIL.UInt64
            self.parent = p
        elif isinstance(p, NumCIL.Complex64.NdArray):
            self.dtype = complex64
            self.cls = NumCIL.Complex64
            self.parent = p
        elif isinstance(p, NumCIL.Complex128.NdArray):
            self.dtype = complex128
            self.cls = NumCIL.Complex128
            self.parent = p
        elif isinstance(p, NumCIL.Boolean.NdArray):
            self.dtype = bool
            self.cls = NumCIL.Boolean
            self.parent = p
        elif isinstance(p, NumCIL.Generic.NdArray[float32]):
            self.dtype = float32
            self.cls = NumCIL.Float
            self.parent = NumCIL.Float.NdArray(p)
        elif isinstance(p, NumCIL.Generic.NdArray[float64]):
            self.dtype = float64
            self.cls = NumCIL.Double
            self.parent = NumCIL.Double.NdArray(p)
        elif isinstance(p, NumCIL.Generic.NdArray[int16]):
            self.dtype = int16
            self.cls = NumCIL.Int16
            self.parent = NumCIL.Int16.NdArray(p)
        elif isinstance(p, NumCIL.Generic.NdArray[uint16]):
            self.dtype = uint16
            self.cls = NumCIL.UInt16
            self.parent = NumCIL.UInt16.NdArray(p)
        elif isinstance(p, NumCIL.Generic.NdArray[int32]):
            self.dtype = int32
            self.cls = NumCIL.Int32
            self.parent = NumCIL.Int32.NdArray(p)
        elif isinstance(p, NumCIL.Generic.NdArray[uint32]):
            self.dtype = uint32
            self.cls = NumCIL.UInt32
            self.parent = NumCIL.UInt32.NdArray(p)
        elif isinstance(p, NumCIL.Generic.NdArray[int64]):
            self.dtype = int64
            self.cls = NumCIL.Int64
            self.parent = NumCIL.Int64.NdArray(p)
        elif isinstance(p, NumCIL.Generic.NdArray[uint64]):
            self.dtype = uint64
            self.cls = NumCIL.UInt64
            self.parent = NumCIL.UInt64.NdArray(p)
        elif isinstance(p, NumCIL.Generic.NdArray[complex64]):
            self.dtype = complex64
            self.cls = NumCIL.Complex64
            self.parent = NumCIL.Complex64.NdArray(p)
        elif isinstance(p, NumCIL.Generic.NdArray[complex128]):
            self.dtype = complex128
            self.cls = NumCIL.Complex128
            self.parent = NumCIL.Complex128.NdArray(p)
        elif isinstance(p, NumCIL.Generic.NdArray[bool]):
            self.dtype = bool
            self.cls = NumCIL.Boolean
            self.parent = NumCIL.Boolean.NdArray(p)
        elif isinstance(p, ndarray):
            self.dtype = p.dtype
            self.cls = p.cls
            self.parent = p.parent
        elif isinstance(p, list) or isinstance(p, tuple):
            self.parent, self.dtype = CreateNdArrayFromList(p)
            self.cls = GetNdClass(self.dtype)
        else:
            raise Exception("Not yet implemented " + str(type(p)))

    def sum(self):
        return self.parent.Sum()

    def max(self):
        return self.parent.Max()

    def min(self):
        return self.parent.Min()

    def repeat(self, repeats, axis = None):
        return self.owncls(self.parent.Repeat(repeats, axis))

    def getsize(self):
        return self.parent.Shape.Elements
    
    size = property(fget=getsize)

    def reshape(self, t):
        if isinstance(t, tuple):
            return self.owncls(self.parent.Reshape(NumCIL.Shape(System.Array[System.Int64](list(t)))))
        else:
            return self.owncls(self.parent.Reshape(NumCIL.Shape(t)))

    def getShapeTuple(self):
        return tuple([int(x.Length) for x in self.parent.Shape.Dimensions])

    def setShapeTuple(self, t):
        self.parent.Reshape(System.Array[System.Int64](list(t)))

    shape = property(fget = getShapeTuple, fset = setShapeTuple)

    def transpose(self):
        if self.parent.Shape.Dimensions.LongLength < 2:
            return self
        else:
            return self.owncls(self.parent.Transpose())

    T = property(fget=transpose)

    def __len__(self):
        return self.parent.Shape.Dimensions[0].Length

    def __getslice__(self, start, end):
        sl = slice(start, end)
        return self.__getitem__(sl)

    def __getitem__(self, slices):
        rval = None
        if isinstance(slices, list) or isinstance(slices, tuple):
            rval = self.owncls(self.parent.Subview(SlicesToRanges(slices), self.collapsedSlicing))
        elif isinstance(slices, slice) or isinstance(slices, int) or isinstance(slices, long) or isinstance(slices, System.Int64) or isinstance(slices, System.Int32) or isinstance(slices, float):
            rval = self.owncls(self.parent.Subview(System.Array[NumCIL.Range]([SliceToRange(slices)]), self.collapsedSlicing))
        else:
            rval = self.owncls(self.parent.Subview(slices, self.collapsedSlicing))

        #If we get a scalar as result, convert it to a python scalar
        if len(rval.shape) == 1 and rval.shape[0] == 1:
            if self.dtype == float32 or self.dtype == float64:
                return float(rval.parent.Value[0])
            else:
                return int(rval.parent.Value[0])
        else:
            return rval

    def __setitem__(self, slices, value):
        v = value
        if isinstance(v, ndarray):
            if v.dtype != self.dtype:
                v = v.astype(self.dtype)
            v = v.parent
        elif isinstance(v, float) or isinstance(v, int):
            c = getattr(self.cls, "NdArray")
            v = c(self.dtype(v))
        elif isinstance(v, list) or isinstance(v, tuple):
            lst = System.Collections.Generic.List[self.dtype]()
            for a in v:
                lst.Add(self.dtype(a))

            c = getattr(self.cls, "NdArray")
            v = c(lst.ToArray())

        if isinstance(slices, list) or isinstance(slices, tuple):
            self.parent.Subview(SlicesToRanges(slices), self.collapsedSlicing).Set(v)
        elif isinstance(slices, slice) or isinstance(slices, long) or isinstance(slices, int) or isinstance(slices, System.Int64) or isinstance(slices, System.Int32):
            self.parent.Subview(System.Array[NumCIL.Range]([SliceToRange(slices)]), self.collapsedSlicing).Set(v)
        else:
            self.parent.Subview(slices, self.collapsedSlicing).Set(v)


    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(other, self)

    def __iadd__(self, other):
        return add(self, other, self)

    def __sub__(self, other):
        return subtract(self, other)

    def __rsub__(self, other):
        return subtract(other, self)

    def __isub__(self, other):
        return subtract(self, other, self)

    def __div__(self, other):
        return divide(self, other)

    def __rdiv__(self, other):
        return divide(other, self)

    def __idiv__(self, other):
        return divide(self, other, self)

    def __mul__(self, other):
        return multiply(self, other)

    def __rmul__(self, other):
        return multiply(other, self)

    def __imul__(self, other):
        return multiply(self, other, self)

    def __mod__(self, other):
        return mod(self, other)

    def __rmod__(self, other):
        return mod(other, self)

    def __imod__(self, other):
        return mod(self, other, self)

    def __pow__(self, other, modulo = None):
        if modulo != None:
            raise Exception("Modulo version of Pow not supported")

        return power(self, other)

    def __rpow__(self, other):
        return power(other, self)

    def __ipow__(self, other, modulo = None):
        if modulo != None:
            raise Exception("Modulo version of Pow not supported")
        power(self, other, self)
        return self

    def __neg__ (self):
        return self.owncls(0 - self.parent)

    def __abs__ (self):
        return self.owncls(self.parent.Abs())

    def __eq__(self, other):
        if other == None:
            return False
        return equal(self, other)

    def __lt__(self, other):
        return lessthan(self, other)

    def __le__(self, other):
        return lessthanorequal(self, other)

    def __gt__(self, other):
        return greaterthan(self, other)

    def __ge__(self, other):
        return greaterthanorequak(self, other)

    def __ne__(self, other):
        if other == None:
            return True
        return notequal(self, other)


    def __str__(self):
        return self.parent.ToString()

    def astype(self, dtype):
        preshape = self.shape
        tmp = self.owncls(clr.Convert(self.parent, GetNdClass(dtype).NdArray))
        # For some reason the type conversion throws away the shape :(
        if tmp.shape != preshape:
            tmp = tmp.reshape(preshape)
        return tmp

    def tofile(self, file, sep="", format="%s"):
        if sep != "":
            raise Exceptio("Only binary output is supported")
        
        tmp = self.parent
        if not tmp.Shape.IsPlain:
            tmp = tmp.Clone()

        NumCIL.Utility.ToFile[self.dtype](tmp.AsArray(), file, tmp.Shape.Offset, tmp.Shape.Elements)


ndarray.owncls = ndarray

def empty(shape, dtype=float, order='C', bohrium=False):
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

    def __init__(self, op, name, nin = 2, nout = 1, nargs = 3):
        self.op = op
        self.name = name
        self.nin = nin
        self.nout = nout
        self.nargs = nargs

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

    def __call__(self, a, b = None, out = None):

        if self.nin == 2 and b == None:
            raise Exception("The operation " + self.name + " requires 2 input operands")
        elif self.nin == 1 and b != None:
            raise Exception("The operation " + self.name + " accepts only 1 input operand")

        cls = None
        outcls = None
        owncls = ndarray
        dtype = float32

        if isinstance(a, ndarray):
            cls = a.cls
            owncls = a.owncls
            dtype = a.dtype
            if isinstance(b, ndarray):
                if a.owncls == matrix or b.owncls == matrix:
                    owncls = matrix
        elif isinstance(b, ndarray):
            cls = b.cls
            owncls = b.owncls
            dtype = b.dtype
        elif out != None and isinstance(out, ndarray):
            cls = out.cls
            owncls = out.owncls
            dtype = out.dtype

        if out != None and isinstance(out, ndarray):
            outcls = out.owncls
        else:
            outcls = owncls

        if cls == None:
            raise Exception("Apply not supported for scalars")
        else:
            f = getattr(cls, self.op)
            if isinstance(a, ndarray):
                a = a.parent
            elif isinstance(a, int) or isinstance(a, long) or isinstance(a, float):
                a = dtype(a)
            if isinstance(b, ndarray):
                b = b.parent
            elif isinstance(b, int) or isinstance(b, long) or isinstance(b, float):
                b = dtype(b)
            if out != None and isinstance(out, ndarray):
                out = out.parent

            if self.nin == 2:
                return outcls(f.Apply(a, b, out))
            else:
                return outcls(f.Apply(a, out))

add = ufunc("Add", "add")
subtract = ufunc("Sub", "subtract")
multiply = ufunc("Mul", "multiply")
divide = ufunc("Div", "divide")
mod = ufunc("Mod", "mod")
maximum = ufunc("Max", "maximum")
minimum = ufunc("Min", "minimum")
abs = ufunc("Abs", "abs", nin = 1, nargs = 2)
exp = ufunc("Exp", "exp", nin = 1, nargs = 2)
log = ufunc("Log", "log", nin = 1, nargs = 2)
sqrt = ufunc("Sqrt", "sqrt", nin = 1, nargs = 2)
rint = ufunc("Round", "rint", nin = 1, nargs = 2)
power = ufunc("Pow", "power")

equal = ufunc("Equal", "equal")
notequal = ufunc("NotEqual", "notequal")
lessthan = ufunc("LessThan", "lessthan")
lessthanorequal = ufunc("LessThanOrEqual", "lessthanorequal")
greaterthan = ufunc("GreaterThan", "greaterthan")
greaterthanorequal = ufunc("GreaterThanOrEqual", "greaterthanorequal")

def array(p):
    return ndarray(p)

class matrix(ndarray):
    collapsedSlicing = False

    def __mul__(self, other):
        if isinstance(other, ndarray):
            return self.owncls(self.parent.MatrixMultiply(other.parent))
        else:
            return self.owncls(self.parent.MatrixMultiply(other))

    def __rmul__(self, other):
        if isinstance(other, ndarray):
            return self.owncls(other.parent.MatrixMultiply(self.parent))
        else:
            return ndarray.__rmul__(self, other)

    def __imul__(self, other):
        if isinstance(other, ndarray):
            self.parent.MatrixMultiply(other.parent, self.parent)
            return self
        else:
            self.parent.MatrixMultiply(other, self.parent)
            return self

    def transpose(self):
        if (len(self.shape) == 1):
            return self.owncls(self.parent.Subview(newaxis, 1))
        else:
            return ndarray.transpose(self)

    T = property(fget=transpose)

matrix.owncls = matrix

def size(x):
    if isinstance(x, ndarray):
        return x.getsize()
    else:
        raise Exception("Can only return size of ndarray")

def concatenate(args, axis=0):
    if args == None or len(args) == 0:
        raise Exception("Invalid args for concatenate")

    cls = None
    dtype = None
    arraycls = None
    for a in args:
        if isinstance(a, ndarray):
            cls = NumCIL.Generic.NdArray[a.dtype] 
            dtype = a.dtype
            arraycls = a.cls

    if cls == None:
        raise Exception("No elements in concatenate were ndarrays")

    arglist = System.Collections.Generic.List[cls]()
    for a in args:
        if isinstance(a, ndarray):
            arglist.Add(a.parent)
        elif isinstance(a, int) or isinstance(a, long) or isinstance(a, float):
            arglist.Add(cls(a))
        else:
            raise Exception("All arguments to concatenate must be ndarrays")

    rargs = arglist.ToArray()
    return ndarray(cls.Concatenate(rargs, axis))

def vstack(args):
    cls = None
    for a in args:
        if isinstance(a, ndarray):
            cls = NumCIL.Generic.NdArray[a.dtype] 

    if cls == None:
        raise Exception("No elements in vstack were ndarrays")

    rargs = []
    for a in args:
        if isinstance(a, float) or isinstance(a, int):
            rargs.append(ndarray(cls(a).Subview(NumCIL.Range.NewAxis, 0)))
        elif isinstance(a, ndarray):
            if a.parent.Shape.Dimensions.LongLength == 1:
                rargs.append(ndarray(a.parent.Subview(NumCIL.Range.NewAxis, 0)))
            else:
                rargs.append(a)
        else:
            raise Exception("Unsupporte element in vstack "  + str(type(a)))

    return concatenate(rargs, 0)

def hstack(args):
    return concatenate(args, 1)

def dstack():
    return concatenate(args, 2)

def shape(el):
    if isinstance(el, ndarray):
        return el.shape
    else:
        raise Exception("Don't know the shape of " + str(type(el)))

class random:
    @staticmethod
    def random(shape, dtype=float, order='C', bohrium=False):
        return ndarray(GetNdClass(dtype).Generate.Random(ReshapeShape(shape)))


def fromfile(file, elems=-1, dtype=float32):
    return ndarray(GetNdClass(dtype).NdArray(NumCIL.Utility.ReadArray[dtype](file, elems)))

def activate_bohrium(active = True):
    try:
        clr.AddReference("NumCIL.Bohrium")
    except Exception:
        raise Exception("Unable to activate NumCIL.Bohrium extensions, make sure that the NumCIL.Bohrium.dll is placed in the same folder as NumCIL.dll")
    
    import NumCIL.Bohrium

    if active:
        NumCIL.Bohrium.Utility.Activate()
    else:
        NumCIL.Bohrium.Utility.Deactivate()
