#region Copyright
/*
This file is part of cphVB and copyright (c) 2012 the cphVB team:
http://cphvb.bitbucket.org

cphVB is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation, either version 3 
of the License, or (at your option) any later version.

cphVB is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the 
GNU Lesser General Public License along with cphVB. 

If not, see <http://www.gnu.org/licenses/>.
*/
#endregion

#define COLLECT_STATS

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Collections.Concurrent;

namespace NumCIL
{
    /// <summary>
    /// A manual implementation of a thread pool
    /// </summary>
    public static class ThreadPool
    {
        /// <summary>
        /// Marker interface for work that can be in the queue
        /// </summary>
        private interface IWorkToken { }

        /// <summary>
        /// This token will shut down a single thread
        /// </summary>
        private class StopToken : IWorkToken { }

        /// <summary>
        /// The token that contains actual work information
        /// </summary>
        private class WorkToken : IWorkToken
        {
            /// <summary>
            /// Constructs a new work token
            /// </summary>
            /// <param name="func">The function to execute</param>
            /// <param name="arg">The argument to pass to the function</param>
            public WorkToken(Action<object> func, object arg)
            {
                this.func = func;
                this.arg = arg;
            }

            /// <summary>
            /// The action to perform
            /// </summary>
            public readonly Action<object> func;
            /// <summary>
            /// The argument input
            /// </summary>
            public readonly object arg;
        }

        /// <summary>
        /// Static initializer
        /// </summary>
        static ThreadPool()
        {
            int i;
            if (int.TryParse(Environment.GetEnvironmentVariable("NUMCIL_MAX_THREADS"), out i))
                m_max_threads = i;
        }

        /// <summary>
        /// The main synchronization unit, a blocking queue of work
        /// </summary>
        private static BlockingCollection<IWorkToken> m_workQueue = new BlockingCollection<IWorkToken>();

        /// <summary>
        /// A counter for the number of threads currently spawned
        /// </summary>
        private static int m_no_threads = 0;
        /// <summary>
        /// A limit on the maximum number of threads that are allowed to be alive
        /// </summary>
        private static int m_max_threads = short.MaxValue;
        /// <summary>
        /// A counter for the number of threads currently active
        /// </summary>
        private static int m_running_threads = 0;
        /// <summary>
        /// A counter for uniquely numbering threads
        /// </summary>
        private static int m_thread_id_counter;

#if COLLECT_STATS
        /// <summary>
        /// The combined number of ticks spent in threads
        /// </summary>
        private static long m_ticksExecuted = 0;

        /// <summary>
        /// Gets the combined number of ticks spent in threads
        /// </summary>
        public static long TicksExecuted { get { return m_ticksExecuted; } set { m_ticksExecuted = value; } }
#endif

        /// <summary>
        /// Adds a work element to the current queue of tasks
        /// </summary>
        /// <param name="callback">The function to execute</param>
        /// <param name="argument">The argument to pass to the function</param>
        public static void QueueUserWorkItem(Action<object> callback, object argument)
        {
            //Make sure we have enough threads to process the request
            if (m_running_threads == m_no_threads)
                NumberOfThreads = Math.Min(MaxThreads, NumberOfThreads + 2);

            m_workQueue.Add(new WorkToken(callback, argument));
        }

        /// <summary>
        /// Gets or sets the number of spawned threads
        /// </summary>
        public static int NumberOfThreads 
        { 
            get { return m_no_threads; } 
            set 
            {
                if (value <= 0)
                    throw new ArgumentOutOfRangeException("NumberOfThreads", "The NumberOfThreads value cannot be less than one");
                if (value > MaxThreads)
                    throw new ArgumentOutOfRangeException("NumberOfThreads", "The NumberOfThreads value cannot be greater than MaxThreads");

                //Start new threads as required
                while (m_no_threads < value)
                {
                    var t = new Thread(ThreadRun);
                    t.IsBackground = true;
                    t.Name = typeof(ThreadPool).FullName + ", thread #" + (m_thread_id_counter++);
                    t.Start(m_workQueue);
                    Interlocked.Increment(ref m_no_threads);
                }

                //Stop threads as required
                int kills = m_no_threads - value;
                for (int i = 0; i < kills; i++)
                {
                    Interlocked.Decrement(ref m_no_threads);
                    m_workQueue.Add(new StopToken());
                }

            } 
        }

        /// <summary>
        /// Stops all current threads, active threads will stop after completion of work.
        /// </summary>
        public static void Reset()
        {
            m_workQueue.CompleteAdding();
            m_workQueue = new BlockingCollection<IWorkToken>();
        }

        /// <summary>
        /// Gets or sets the maximum number of threads allowed.
        /// </summary>
        public static int MaxThreads 
        { 
            get { return m_max_threads; } 
            set 
            {
                if (value <= 0)
                    throw new ArgumentOutOfRangeException("MaxThreads", "The MaxThreads value cannot be less than one");
                m_max_threads = value;

                if (NumberOfThreads > MaxThreads)
                    NumberOfThreads = MaxThreads;
            } 
        }

        /// <summary>
        /// The primary thread implmentation
        /// </summary>
        private static void ThreadRun(object _queue)
        {
            bool failureExit = true;
            BlockingCollection<IWorkToken> queue = (BlockingCollection<IWorkToken>)_queue;
            try
            {
                while (true)
                {
                    //Grab a work token
                    IWorkToken token = null;
					try { token = queue.Take(); }
					catch (InvalidOperationException) 
					{
						//Upon completion the queue will throw InvalidOperationException
						if (!queue.IsCompleted) 
							throw;
					}

                    if (token == null || token is StopToken)
                    {
                        if (token is StopToken)
                            failureExit = false;
                        return;
                    }

#if COLLECT_STATS
                    long ticks = DateTime.Now.Ticks;
#endif
                    try
                    {
                        //Do the work
                        Interlocked.Increment(ref m_running_threads);
                        WorkToken work = (WorkToken)token;
                        work.func(work.arg);
                    }
                    catch (ThreadAbortException)
                    {
                        Thread.ResetAbort();
                    }
                    finally
                    {
                        Interlocked.Decrement(ref m_running_threads);
#if COLLECT_STATS
                        Interlocked.Add(ref m_ticksExecuted, DateTime.Now.Ticks - ticks);
#endif
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("A thread crashed!!!{0}{1}", Environment.NewLine, ex.ToString());
            }
            finally
            {
                if (failureExit)
                    Interlocked.Decrement(ref m_no_threads);
            }
        }
    }
}
