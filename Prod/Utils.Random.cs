using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace SharpNet
{
    public static partial class Utils
    {
        public static void UniformDistribution(Span<float> toRandomize, Random rand, double minValue, double maxValue)
        {
            for (int j = 0; j < toRandomize.Length; ++j)
            {
                toRandomize[j] = (float)(minValue + rand.NextDouble() * (maxValue - minValue));
            }
        }
        public static void UniformDistribution(Span<byte> toRandomize, Random rand, byte minValue, byte maxValue)
        {
            for (int j = 0; j < toRandomize.Length; ++j)
            {
                toRandomize[j] = (byte)(minValue + rand.Next(maxValue - minValue + 1));
            }
        }
        public static void NormalDistribution(Span<float> toRandomize, Random rand, double mean, double stdDev)
        {
            for (int j = 0; j < toRandomize.Length; ++j)
            {
                toRandomize[j] = (float)NextDoubleNormalDistribution(rand, mean, stdDev);
            }
        }
        public static void UniformDistribution(Span<int> toRandomize, Random rand, int minValue, int maxValue)
        {
            for (int j = 0; j < toRandomize.Length; ++j)
            {
                toRandomize[j] = rand.Next(minValue, maxValue+1);
            }
        }
        public static void Shuffle<T>(IList<T> list, Random rand)
        {
            int n = list.Count;
            while (n > 1)
            {
                n--;
                int k = rand.Next(n + 1);
                (list[k], list[n]) = (list[n], list[k]);
            }
        }
        public static void Shuffle<T>(IList<T> list, Random rand, int blockSize)
        {
            Debug.Assert(list.Count%blockSize == 0);
            var blockIds =Enumerable.Range(0, list.Count / blockSize).ToList();
            Shuffle(blockIds, rand);
            var listCopy = new List<T>(list);
            foreach (var t in blockIds)
            {
                for (int j = 0; j < blockSize; ++j)
                {
                    list[t*blockSize + j] = listCopy[t*blockSize + j];
                }
            }
        }
        public static double BetaDistribution(double a, double b, Random rand)
        {
            var alpha = a + b;
            double beta;
            if (Math.Min(a, b) <= 1.0)
            {
                beta = Math.Max(1 / a, 1 / b);
            }
            else
            {
                beta = Math.Sqrt(alpha - 2.0) / (2 * a * b - alpha);
            }

            double gamma = a + 1 / beta;
            double w;
            while (true)
            {
                var u1 = rand.NextDouble();
                var u2 = rand.NextDouble();
                var v = beta * Math.Log(u1 / (1 - u1));
                w = a * Math.Exp(v);
                var tmp = Math.Log(alpha / (b + w));
                if ((alpha * tmp + (gamma * v) - 1.3862944) >= Math.Log(u1 * u1 * u2))
                {
                    break;
                }
            }

            return w / (b + w);
        }
        private static double NextDoubleNormalDistribution(Random rand, double mean, double stdDev)
        {
            //uniform(0,1) random double
            var u1 = rand.NextDouble();
            //uniform(0,1) random double
            var u2 = rand.NextDouble();
            //random normal(0,1)
            var randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            //random normal(mean,stdDev^2)
            return mean + stdDev * randStdNormal;
        }
        /// <summary>
        /// make a random coin flip, and returns:
        ///     true if head
        ///     false if tail
        /// </summary>
        /// <returns></returns>
        public static bool RandomCoinFlip()
        {
            return new Random(RandomSeed()).NextDouble() > 0.5;
        }

        public static int RandomSeed()
        {
            var randomSeed = Guid.NewGuid().GetHashCode();
            return randomSeed;
        }
    }
}
