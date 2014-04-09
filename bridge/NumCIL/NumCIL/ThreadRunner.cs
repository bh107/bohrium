#define COUNT_TICKS

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;

namespace NumCIL
{
    /// <summary>
    /// Class that simplifies running parallel tasks
    /// </summary>
    public class ThreadRunner : IDisposable
    {
        /// <summary>
        /// The list of active threads
        /// </summary>
        private Thread[] m_thread;
        /// <summary>
        /// The barrier used to wait for starting
        /// </summary>
        private Barrier m_startBarrier;
        /// <summary>
        /// The barrier used to wait for completion
        /// </summary>
        private Barrier m_finishBarrier;
        /// <summary>
        /// The action currently being executed
        /// </summary>
        private Action<int> m_action;
        /// <summary>
        /// A flag indicating if this instance is now disposed
        /// </summary>
        private volatile bool m_disposed = false;
        /// <summary>
        /// A counter for the number of ticks executed in the threads
        /// </summary>
        private static long _ticks = 0;
        
        /// <summary>
        /// The maximum number of milliseconds to wait for work
        /// before checking the dispose flag
        /// </summary>
        private static int m_finalizeGuardTimeout = (int)TimeSpan.FromSeconds(60).TotalMilliseconds;

        /// <summary>
        /// Constructs a new instance of the ThreadParallel class
        /// </summary>
        /// <param name="count">The number of parallel threads</param>
        public ThreadRunner(int count)
        {
            ResizePool(count);
        }

        /// <summary>
        /// Resizes the thread count
        /// </summary>
        /// <param name="count">The number of threads to have</param>
        private void ResizePool(int count)
        {
            if (m_thread != null)
            {
                this.Dispose();
                m_disposed = false;
            }

            m_startBarrier = new Barrier(count + 1);
            m_finishBarrier = new Barrier(count + 1);
            m_thread = new Thread[count];
            for (var i = 0; i < count; i++)
            {
                m_thread[i] = new Thread(ThreadExecute) {
                    Name = "ThreadParallel Worker #" + i, 
                    IsBackground = true
                };
                m_thread[i].Start(i);
            }
        }

        /// <summary>
        /// Gets or sets the total number of ticks used in threads
        /// </summary>
        public static long TicksExecuted
        {
            get { return _ticks; }
            set { _ticks = value; }
        }


        /// <summary>
        /// Gets or sets the number of threads
        /// </summary>
        public int Threads
        {
            get { return m_thread.Length; }
            set
            {
                if (value == m_thread.Length)
                    return;

                ResizePool(value);
            }
        }

        /// <summary>
        /// Thead worker function
        /// </summary>
        /// <param name="data">The thread index</param>
        private void ThreadExecute(object data)
        {
            var index = (int)data;
            while (!m_disposed)
            {
				bool work = m_startBarrier.SignalAndWait(m_finalizeGuardTimeout);
				if (m_disposed)
					return;
				if (!work) //In case we woke up due to inactivity
               		continue;
               	if (m_action == null) //Should not happen, but it does :(
               		continue;
#if COUNT_TICKS
                long ticks = DateTime.Now.Ticks;
#endif
                m_action.Invoke(index);
#if COUNT_TICKS
                Interlocked.Add(ref _ticks, DateTime.Now.Ticks - ticks);
#endif
                m_finishBarrier.SignalAndWait();
            }
        }

        /// <summary>
        /// Executes the given action on all threads, passing each thread id to the function
        /// </summary>
        /// <param name="action">The action to run</param>
        public void RunParallel(Action<int> action)
        {
            if (m_disposed)
                throw new ObjectDisposedException("");

            m_action = action;
            m_startBarrier.SignalAndWait();
            m_finishBarrier.SignalAndWait();
            m_action = null;
        }

        /// <summary>
        /// Disposes all resources associated with the instance
        /// </summary>
        public void Dispose()
        {
			Dispose(true);
        }
        
        /// <summary>
        /// Disposes the resources and shuts down the threadpool.
        /// </summary>
        /// <param name="disposing">Set to <c>true</c> if disposing, false otherwise.</param>
        protected void Dispose(bool disposing)
        {
            if (!m_disposed)
            {
                m_disposed = true;
                m_startBarrier.RemoveParticipant();
                m_finishBarrier.RemoveParticipant();
				
                if (disposing)
                    GC.SuppressFinalize(this);
				
                var failed = 0;
                foreach (var t in m_thread)
                {
                    t.Join(TimeSpan.FromSeconds(10));
                    if (t.IsAlive)
                        failed++;
                }
				
                if (failed != 0)
					throw new Exception(string.Format("Failed to shut down {0} threads", failed));
			}
		}
        
        /// <summary>
        /// Releases unmanaged resources and performs other cleanup operations before the
        /// <see cref="NumCIL.ThreadRunner"/> is reclaimed by garbage collection.
        /// </summary>
        ~ThreadRunner()
        {
        	Dispose(false);
        }
    }
}
