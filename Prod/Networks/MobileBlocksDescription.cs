using System;
using System.Collections.Generic;

namespace SharpNet.Networks
{
    public class MobileBlocksDescription
    {
        #region public properties
        public int KernelSize { get; }
        public int NumRepeat { get; }
        public int OutputFilters { get; }
        public int ExpandRatio { get; }
        public bool IdSkip { get; }
        public int RowStride { get; }
        public int ColStride { get; }
        public double SeRatio { get; }
        #endregion

        private MobileBlocksDescription(int kernelSize, int numRepeat, int outputFilters, int expandRatio, bool idSkip, int rowStride, int colStride, double seRatio)
        {
            KernelSize = kernelSize;
            NumRepeat = numRepeat;
            OutputFilters = outputFilters;
            ExpandRatio = expandRatio;
            IdSkip = idSkip;
            RowStride = rowStride;
            ColStride = colStride;
            SeRatio = seRatio;
        }

        public static List<MobileBlocksDescription> Default()
        {
            return new List<MobileBlocksDescription>
            {
                new (3, 1, 16, 1, true, 1, 1, .25),
                new (3, 2, 24, 6, true, 2, 2, .25),
                new (5, 2, 40, 6, true, 2, 2, .25),
                new (3, 3, 80, 6, true, 2, 2, .25),
                new (5, 3, 112, 6, true, 1, 1, .25),
                new (5, 4, 192, 6, true, 2, 2, .25),
                new (3, 1, 320, 6, true, 1, 1, .25)
            };
        }

        public MobileBlocksDescription ApplyScaling(float widthCoefficient, int depthDivisor, float depthCoefficient)
        {
            return new MobileBlocksDescription(
                KernelSize,
                RoundNumRepeat(NumRepeat, depthCoefficient), 
                RoundFilters(OutputFilters, widthCoefficient, depthDivisor),
                ExpandRatio, 
                IdSkip, 
                RowStride, 
                ColStride, 
                SeRatio);
        }

        public MobileBlocksDescription WithStride(int newRowStride, int newColStride)
        {
            return new MobileBlocksDescription(
                KernelSize,
                NumRepeat,
                OutputFilters,
                ExpandRatio,
                IdSkip,
                newRowStride,
                newColStride,
                SeRatio);
        }

        public static int RoundNumRepeat(int numRepeat, float depthCoefficient)
        {
            return (int)Math.Ceiling(depthCoefficient * numRepeat);
        }

        public static int RoundFilters(int outputFilters, float widthCoefficient, int depthDivisor)
        {
            float tmpFilters = outputFilters* widthCoefficient;
            // ReSharper disable once PossibleLossOfFraction
            float newFilters = ((int)((tmpFilters + depthDivisor / 2) / depthDivisor)) * depthDivisor;
            newFilters = Math.Max(depthDivisor, newFilters);
            //Make sure that round down does not go down by more than 10%.
            if (newFilters < 0.9 * tmpFilters)
            {
                newFilters += depthDivisor;
            }
            return (int)newFilters;
        }

    }
}