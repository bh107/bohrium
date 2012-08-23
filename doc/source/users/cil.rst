Common Intermediate Language (CIL)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The purpose of NumCIL is to provide a library that provides any CIL language with an n-dimensional array class and perform operations on these. Using the notion of a n-dimensional array, it becomes possible to express many common scientific computations.

The NdArray in NumCIL work like the ndarray from NumPy. You can slice, transpose, broadcast elements. You can run your programs entirely in CIL or use cphVB to offload computation. If you include the NumCIL.Unsafe.dll in your project, NumCIL will seamlessly optimize computation in a manner not normally allowed by CIL. Including the dll will usually make your program run faster, and will just be disabled if your security settings prevent this.

To use cphVB to do the computation, you need to include the NumCIL.cphVB.dll, as well as have a working setup of cphVB (see the installation section). After this, simply call NumCIL.cphVB.Utility.Activate() as one of the first lines of your program and all execution will be performed with cphVB. If cphVB is not correctly installed, the call will generate an error.

Due to the cross-language features of the CIL, any program in any .Net/Mono supported language can use NumCIL and cphVB. Below are some of the commonly found CIL languages, but usage is similar for other languages.

Basic usage
------------

To use NumCIL in a project, simply include a reference to the NumCIL.dll. You can then write some code:
	
C#::

    using NumCIL.Double;
    
    namespace Test
    {
        public static class Program
        {
            static void Main(string[] args)
            {
                //Make a vector of length 3
                var a = Generate.Ones(3);
                //Add a scalar value to each element
                var b = a + 20;
                //Writes: [21, 21, 21]
                System.Console.WriteLine(b);

                //Multiply in-place
                Mul.Apply(-2, b, b);
                //Get absolute values
                var c = b.Abs();
                //Writes: [42, 42, 42]
                System.Console.WriteLine(c);
            }
        }
    }
	

F#::

	module main
	open NumCIL.Double
	
	//Make a vector of length 3
	let a = Generate.Ones([| 3L |])
	//Add a scalar value to each element
	let b = a + 20.0
	//Writes: [21, 21, 21]
	printfn "%O" b
	
	//Multiply in-place
	Mul.Apply(-2.0, b, b) |> ignore
	//Get absolute values
	let c = b.Abs()
	//Writes: [42, 42, 42]
	printfn "%O" c
	
IronPython::

	import numcil as np
		
	#Make a vector of length 3
	a = np.ones((3,), dtype=float)
	#Add a scalar value to each element
	b = a + 20
	#Writes: [21, 21, 21]
	print b
	
	#In-place multiply
	b *= -2
	#Get absolute values
	c = np.abs(b)
	#Writes: [42, 42, 42]
	print c	


The using statement for C# and F# imports some helpfull methods, and also sets the data precision. In this case we use NumCIL.Double, so all computation will be done with 64bit floating point numbers. You can mix types or access the types directly, like NumCIL.Int8.Generate.Ones(3) if you prefer.

In IronPython you can also access the NumCIL types directly, but the ndarray type exposes the underlying library in a more convenient way that mimics the numpy package.


View usage
--------------

In NumCIL, you can reshape your data views through indexing. This is a powerfull feature that allows you to operate on subsets of your data as well as combine different parts of the data. An example tells how this works:

C#::

    using NumCIL.Double;
    using R = NumCIL.Range;
    
    namespace Test
    {
        public static class Program
        {
            static void Main(string[] args)
            {
                //Make a 2x3 array
                var a = Generate.Ones(new long[] { 2, 3 });
                //Make a vector with 2 elements from an array
                var b = new NdArray(new double[] { 1, -2, 2 });
                //Make the numbers for 0 to 9
                var c = Generate.Arange(10);
    
                //Reshape the array to a 3x3 array
                c = c.Reshape(new long[] { 3, 3 });
    
                //Index the array to exclude an element,
                // so we obtain a 3x2 array
                c = c[R.All, R.Slice(0, -1)];
    
                //Transpose this to make it compatible with a
                c = c.Transpose();
    
                //b is implicitly broadcasted to:
                //b[R.New, R.All], with first dimension
                //replicated to create a 2x3 array
    
                var d = a + b + c;
                //Writes: [[2 2 9], [3 3 10]]
                System.Console.WriteLine(d);
            }
        }
    }
	
F#::

	module main
	open NumCIL.Double
	open NumCIL
	
	//Make a 2x3 array
	let a = Generate.Ones([| 2L; 3L |])
	//Make a vector with 2 elements from an array
	let b = NdArray([|1.0; -2.0; 2.0|])
	//Make the numbers for 0 to 9
	let numbers = Generate.Arange(10L)
	
	//Reshape the array to a 3x3 array
	let csquare = numbers.Reshape(Shape([| 3L; 3L |]))
	
	//Index the array to exclude an element,
	// so we obtain a 3x2 array
	let cpart = csquare.Subview([| Range.All; Range.Slice(0L, -1L) |], true)
	
	//Transpose this to make it compatible with a
	let c = cpart.Transpose();
	
	//b is implicitly broadcasted to:
	//b[R.New, R.All], with first dimension
	//replicated to create a 2x3 array
	
	let d = a + b + c
	//Writes: [[2 2 9], [3 3 10]]
	printfn "%O" d

IronPython::

	import numcil as np
	
	#Make a 2x3 array
	a = np.ones((2, 3), dtype=float)
	#Make a vector with 2 elements from an array
	b = np.ndarray((1, -2, 2)).astype(float)
	#Make the numbers for 0 to 9
	c = np.arange(10L)
	
	#Reshape the array to a 3x3 array
	c = c.reshape((3, 3))
	
	#Index the array to exclude an element,
	# so we obtain a 3x2 array
	c = c[:, 0:-1]
	
	#Transpose this to make it compatible with a
	c = c.transpose()
	
	#b is implicitly broadcasted to:
	#b[R.New, R.All], with first dimension
	#replicated to create a 2x3 array
	
	d = a + b + c
	#Writes: [[2 2 9], [3 3 10]]
	print d

