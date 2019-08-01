using System;
using System.Runtime.InteropServices;

namespace SharpNet.Data
{
    [StructLayout(LayoutKind.Sequential)]
    public struct size_t
    {
        private readonly UIntPtr value;

        public size_t(ulong value)
        {
            this.value = new UIntPtr(value);
        }
        public static implicit operator size_t(ulong t)
        {
            return new size_t(t);
        }
        public static implicit operator ulong(size_t t)
        {
            return t.value.ToUInt64();
        }

        public override string ToString()
        {
            return ((ulong)this).ToString();
        }
    }
}
