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
                m_startBarrier.SignalAndWait();
#if COUNT_TICKS
                long ticks = DateTime.Now.Ticks;
#endif
                if (m_disposed)
                    return;
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
            if (!m_disposed)
            {
                m_disposed = true;
                m_startBarrier.RemoveParticipant();
                m_finishBarrier.RemoveParticipant();

                var ok = true;
                foreach (var t in m_thread)
                    ok &= t.Join(TimeSpan.FromSeconds(10));

                if (!ok)
                    throw new Exception("Failed to shut down threads?");
            }
        }
    }
}
