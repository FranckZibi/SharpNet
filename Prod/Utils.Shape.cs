using System;
using System.Linq;

namespace SharpNet
{
    public static partial class Utils
    {
        public static int[] CloneShapeWithNewCount(int[] shape, int newCount)
        {
            if (shape == null)
            {
                return null;
            }
            var result = (int[])shape.Clone();
            result[0] = newCount;
            return result;
        }
        public static long LongProduct(int[] data)
        {
            return LongProduct(data.Select(i=>(long)i).ToArray());
        }

        private static long LongProduct(long[] data)
        {
            if ((data == null) || (data.Length == 0))
            {
                return 0;
            }

            long result = data[0];
            for (int i = 1; i < data.Length; ++i)
            {
                result *= data[i];
            }

            return result;
        }
        public static int Product(int[] data)
        {
            if ((data == null) || (data.Length == 0))
            {
                return 0;
            }

            var result = data[0];
            for (int i = 1; i < data.Length; ++i)
            {
                result *= data[i];
            }

            return result;
        }
        public static string ShapeToStringWithBatchSize(int[] shape)
        {
            if (shape == null)
            {
                return "(?)";
            }

            return "(None, " + string.Join(", ", shape.Skip(1)) + ")";
        }
        public static string ShapeToString(int[] shape)
        {
            if (shape == null)
            {
                return "(?)";
            }

            return "(" + string.Join(", ", shape) + ")";
        }
    }
}
