using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Tester
{
    /// <summary>Implementation of a High Performance timer, that falls back to DateTime.Ticks if not present</summary>
    public class CustomTimer
    {
        [System.Runtime.InteropServices.DllImport("Kernel32.dll")]
        private static extern bool QueryPerformanceCounter(out long lpPerformanceCount);
        [System.Runtime.InteropServices.DllImport("Kernel32.dll")]
        private static extern bool QueryPerformanceFrequency(out long lpFrequency);
        private long startTime = 0;
        private long stopTime = 0;
        private long freq = TimeSpan.TicksPerSecond;
        private bool mHasHighRes = false;

        public CustomTimer()
        {
            try
            {
                if (QueryPerformanceFrequency(out freq))
                    mHasHighRes = true;
            }
            catch { }
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

    public class DispTimer : IDisposable
    {
        private CustomTimer m_timer;
        private string m_activity = "Timer";

        public DispTimer()
        {
            m_timer = new CustomTimer(); 
            m_timer.Start();
        }

        public DispTimer(string activity)
            : this()
        {
            m_activity = activity;
        }

        protected virtual void ReportTime(double seconds)
        {
            Console.WriteLine("{0}: {1}", m_activity, seconds);
        }

        #region IDisposable Members

        public void Dispose()
        {
            if (m_timer != null)
            {
                m_timer.Stop();
                ReportTime(m_timer.Duration);
                m_timer = null;
            }
        }

        #endregion
    }
}
