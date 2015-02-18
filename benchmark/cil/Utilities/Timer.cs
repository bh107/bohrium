#region Copyright
/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium:
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
using System.Text;

namespace Utilities
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
            Console.WriteLine("{0} elapsed-time: {1}", m_activity, seconds);
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

