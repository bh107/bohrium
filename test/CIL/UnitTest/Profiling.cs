#region Copyright
/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium
team <http://www.bh107.org>.

Bohrium is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation, either version 3 
of the License, or (at your option) any later version.

Bohrium is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the 
GNU Lesser General Public License along with Bohrium. 

If not, see <http://www.gnu.org/licenses/>.
*/
#endregion

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Linq.Expressions;

namespace UnitTest
{
    using T = System.Double;

    static class Profiling
    {
        #region Iteration duration values
        private const long ITERATIONS = 1000000000;
        private const long REPETITIONS = 10;
        private const T INCR_VALUE = 1;
        #endregion

        #region Lambda functions and structures
        //Basic lambda function
        private static readonly Func<T, T, T> lambda_add = (a, b) => a + b;
        //Holder for lambda function that is compiled dynamically
        private static readonly Func<T, T, T> dynamic_lambda_add;

        //Definition of the IAdd interface
        private interface IAdd { T Add(T a, T b); }

        //Shared copy of the statically instanciated vAdd class
        private static readonly IAdd virtual_add;

        //A struct and a class that implements the IAdd interface
        private class vAdd : IAdd { public T Add(T a, T b) { return a + b; } }
        private struct sAdd : IAdd { public T Add(T a, T b) { return a + b; } }

        //The indirection interface
        private interface IAdd2 { T Add(T a); }
        private struct sAdd2<C> : IAdd2 where C : IAdd
        {
            private T m_value;
            private C m_op;

            public sAdd2(T value, C op) { m_value = value; m_op = op; }
            public T Add(T a) { return m_op.Add(m_value, a); } 
        }

        #endregion

        //Static initialization of the lambda function + creation of the class instance
        static Profiling()
        {
            ParameterExpression paramA = Expression.Parameter(typeof(T), "a"),
                                paramB = Expression.Parameter(typeof(T), "b");

            BinaryExpression body = Expression.Add(paramA, paramB);
            dynamic_lambda_add = Expression.Lambda<Func<T, T, T>>(body, paramA, paramB).Compile();

            virtual_add = new vAdd();
        }

        public static void RunProfiling()
        {
#if DEBUG
            Console.WriteLine("You are running in DEBUG mode, results are unreliable");
#endif
            List<Func<T>> funcs = new List<Func<T>>();
            funcs.Add(TestDirect);
            funcs.Add(TestInlineable);
            funcs.Add(TestStaticLambda);
            funcs.Add(TestLocalLambda);
            funcs.Add(TestCompiledLambda);
            funcs.Add(TestVirtualFunctionCall);
            funcs.Add(TestStructConstraint);
            funcs.Add(TestStructConstraintInstance);
            funcs.Add(TestStructConstraintIndirection);

            HiPerfTimer timer = new HiPerfTimer();

            foreach (Func<T> k in funcs)
            {
                Console.WriteLine("Profiling " + k.Method.Name + " with " + typeof(T).FullName + " ...");
                double[] times = new double[REPETITIONS];
                for (int i = 0; i < REPETITIONS; i++)
                {
                    timer.Start();
                    k();
                    timer.Stop();
                    times[i] = timer.Duration;
                }

                Console.WriteLine("Avg: {0},\tMax: {1}\tMin: {2}",
                    times.Sum() / REPETITIONS,
                    times.Max(),
                    times.Min()
                );
                Console.WriteLine();
            }
        }

        private static T TestInlineable()
        {
            //Should be as fast as the direct because the Add(...) call is inlined
            T start = 0;
            for (long i = 0; i < ITERATIONS; i++)
                start = Add(start, INCR_VALUE);

            return start;
        }

        private static T TestDirect()
        {
            //Should be the fastest version, no tricks, just plain addition
            T start = 0;
            for (long i = 0; i < ITERATIONS; i++)
                start = start + INCR_VALUE;

            return start;
        }

        private static T TestStaticLambda()
        {
            //Simple version, using a statically defined lambda op
            T start = 0;
            for (long i = 0; i < ITERATIONS; i++)
                start = lambda_add(start, INCR_VALUE);

            return start;
        }

        private static T TestLocalLambda()
        {
            //Simple version, using a local lambda op
            Func<T, T, T> op = (a, b) => a + b;
            T start = 0;
            for (long i = 0; i < ITERATIONS; i++)
                start = op(start, INCR_VALUE);

            return start;
        }

        private static T TestCompiledLambda()
        {
            //Using the dynamic compiled lambda op
            T start = 0;
            for (long i = 0; i < ITERATIONS; i++)
                start = dynamic_lambda_add(start, INCR_VALUE);

            return start;
        }


        //Helper function to use correct function interface
        private static T TestStructConstraint()
        {
            return TestStructConstraint_Inner<sAdd>();
        }

        private static T TestStructConstraint_Inner<C>() where C : struct, IAdd
        {
            //This is bound to the sAdd struct, not the interface,
            // so the invocations below are not neccesarily virtual
            // function calls and can be inlined
            C localAdd = new C();

            T start = 0;
            for (long i = 0; i < ITERATIONS; i++)
                start = localAdd.Add(start, INCR_VALUE);

            return start;

        }

        //Helper function to use correct function interface
        private static T TestStructConstraintInstance()
        {
            return TestStructConstraintInstance_Inner<sAdd>(new sAdd());
        }

        private static T TestStructConstraintInstance_Inner<C>(C op) where C : struct, IAdd
        {
            //op is bound to the sAdd struct, not the interface,
            // so the invocations below are not neccesarily virtual
            // function calls and can be inlined

            T start = 0;
            for (long i = 0; i < ITERATIONS; i++)
                start = op.Add(start, INCR_VALUE);

            return start;

        }

        //Helper function to use correct function interface
        private static T TestStructConstraintIndirection()
        {
            return TestStructConstraintIndirection_Inner<sAdd2<sAdd>>(new sAdd2<sAdd>(INCR_VALUE, new sAdd()));
        }

        private static T TestStructConstraintIndirection_Inner<C>(C op) where C : struct, IAdd2
        {
            //op is bound to the sAdd struct, not the interface,
            // so the invocations below are not neccesarily virtual
            // function calls and can be inlined

            T start = 0;
            for (long i = 0; i < ITERATIONS; i++)
                start = op.Add(start);

            return start;

        }

        private static T TestVirtualFunctionCall()
        {
            //Since we are accessing the Add call through an interface, each call is virtual and thus not inlined
            T start = 0;
            for (long i = 0; i < ITERATIONS; i++)
                start = virtual_add.Add(start, INCR_VALUE);

            return start;
        }

        //A simple inline-able function
        private static T Add(T a, T b)
        {
            return a + b;
        }

        /// <summary>Implementation of a High Performance timer, that falls back to DateTime.Ticks if not present</summary>
        private class HiPerfTimer
        {
            [System.Runtime.InteropServices.DllImport("Kernel32.dll")]
            private static extern bool QueryPerformanceCounter(out long lpPerformanceCount);
            [System.Runtime.InteropServices.DllImport("Kernel32.dll")]
            private static extern bool QueryPerformanceFrequency(out long lpFrequency);
            private long startTime = 0;
            private long stopTime = 0;
            private long freq = TimeSpan.TicksPerSecond;
            private bool mHasHighRes = false;

            public HiPerfTimer()
            {
                try
                {
                    if (QueryPerformanceFrequency(out freq))
                        mHasHighRes = true;
                }
                catch { }

                Console.WriteLine("HiPerf timer {0} supported, using {1} ticks/second", mHasHighRes ? "IS" : "NOT", freq);

            }

            /// <summary>Starts the timer</summary>
            public void Start()
            {
                if (mHasHighRes)
                    QueryPerformanceCounter(out startTime);
                else
                    startTime = DateTime.Now.Ticks;
            }
            /// <summary>Stops the timer</summary>
            public void Stop()
            {
                if (mHasHighRes)
                    QueryPerformanceCounter(out stopTime);
                else
                    stopTime = DateTime.Now.Ticks;
            }

            /// <summary>Gets duration in seconds</summary>
            public double Duration { get { return (double)(stopTime - startTime) / (double)freq; } }
        }
    }
}
