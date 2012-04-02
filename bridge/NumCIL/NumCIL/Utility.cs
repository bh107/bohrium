using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NumCIL.Generic;

namespace NumCIL
{
    public static class Utility
    {
        //TODO: Remove this class because it uses lambda functions which compile into virtual function calls, or modify the IL to generate a non-virtual function call type

        /// <summary>
        /// Simple helper class that can make any index-able class enumerable
        /// </summary>
        /// <typeparam name="T">The value type that is being enumerated</typeparam>
        public class AnyEnumerator<T> : IEnumerator<T>
        {
            private Func<long, T> m_get;
            private long m_size;
            private long m_index = -1;

            public AnyEnumerator(Func<long, T> get, long size)
            {
                m_get = get;
                m_size = size;
            }

            #region IEnumerator<T> Members

            public T Current
            {
                get { return m_get(m_index); }
            }

            #endregion

            #region IDisposable Members

            public void Dispose()
            {
            }

            #endregion

            #region IEnumerator Members

            object System.Collections.IEnumerator.Current
            {
                get { return m_get(m_index); }
            }

            public bool MoveNext()
            {
                m_index++;
                return m_index < m_size;
            }

            public void Reset()
            {
                m_index = 0;
            }

            #endregion
        }

        /// <summary>
        /// An extension method that enables the use of LinQ on a plain IEnumerable, which is used extensively in Mono.Cecil
        /// </summary>
        /// <typeparam name="T">The type that the IEnumerable contains</typeparam>
        public class TypedEnumerable<T> : IEnumerable<T>
        {
            private System.Collections.IEnumerable col;

            public TypedEnumerable(System.Collections.IEnumerable col) { this.col = col; }

            #region IEnumerable<Instruction> Members
            public IEnumerator<T> GetEnumerator() { return new TypedEnumerator(this.col.GetEnumerator()); }
            #endregion

            #region IEnumerable Members
            System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() { return this.col.GetEnumerator(); }
            #endregion

            private class TypedEnumerator : IEnumerator<T>
            {
                private System.Collections.IEnumerator m_base;
                public TypedEnumerator(System.Collections.IEnumerator @base) { m_base = @base; }
                #region IEnumerator<T> Members
                public T Current { get { return (T)m_base.Current; } }
                #endregion

                #region IDisposable Members
                public void Dispose()
                {
                    if (m_base is IDisposable)
                        ((IDisposable)m_base).Dispose();
                }
                #endregion

                #region IEnumerator Members
                object System.Collections.IEnumerator.Current { get { return m_base.Current; } }
                public bool MoveNext() { return m_base.MoveNext(); }
                public void Reset() { m_base.MoveNext(); }
                #endregion
            }
        }

        public static void Apply<T>(this IEnumerable<T> seq, Action<T> a)
        {
            foreach (T n in seq)
                a(n);
        }
    }
}
