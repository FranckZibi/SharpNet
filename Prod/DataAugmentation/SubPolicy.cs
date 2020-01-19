using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using SharpNet.CPU;
using SharpNet.DataAugmentation.Operations;

namespace SharpNet.DataAugmentation
{
    public static class SubPolicy
    {
        public static void Apply(List<Operation> subPolicy,
            int indexInMiniBatch,
            CpuTensor<float> xInputMiniBatch,
            CpuTensor<float> xOutputMiniBatch,
            CpuTensor<float> yMiniBatch,
            Func<int, int> indexInMiniBatchToCategoryIndex,
            ImageDataGenerator.FillModeEnum fillMode)
        {
            Debug.Assert(xInputMiniBatch.SameShape(xOutputMiniBatch));
            if (subPolicy.Count == 0)
            {
                xInputMiniBatch.CopyTo(xOutputMiniBatch);
                return;
            }
            var nbRows = xInputMiniBatch.Shape[2];
            var nbCols = xInputMiniBatch.Shape[3];
            var previous = (subPolicy.Count % 2 == 1)? new CpuTensor<float>(xOutputMiniBatch.Shape, "buffer"): xOutputMiniBatch;
            var next = (subPolicy.Count % 2 == 1)? xOutputMiniBatch:new CpuTensor<float>(xOutputMiniBatch.Shape, "buffer");
            double unconvertCX =0, unconvertAX = 0, unconvertBX = 0, unconvertCY = 0, unconvertAY = 0, unconvertBY = 0;

            for (int i = 0; i < subPolicy.Count; ++i)
            {
                var policy = subPolicy[i];
                var policyMayChangeCoordinates = policy.ChangeCoordinates();
                
                if (policyMayChangeCoordinates)
                {
                    unconvertCX = policy.Unconvert_Slow(0, 0).row;
                    unconvertAX = policy.Unconvert_Slow(1, 0).row - unconvertCX;
                    unconvertBX = policy.Unconvert_Slow(0, 1).row - unconvertCX;
                    unconvertCX += 1e-8;

                    unconvertCY = policy.Unconvert_Slow(0, 0).col;
                    unconvertAY = policy.Unconvert_Slow(1, 0).col - unconvertCY;
                    unconvertBY = policy.Unconvert_Slow(0, 1).col - unconvertCY;
                    unconvertCY += 1e-8;
                }

                for (int channel = 0; channel < xInputMiniBatch.Shape[1]; ++channel)
                {
                    for (int rowOutput = 0; rowOutput < nbRows; ++rowOutput)
                    {
                        var outputPictureIdx = next.Idx(indexInMiniBatch, channel, rowOutput, 0);
                        for (int colOutput = 0; colOutput < nbCols; ++colOutput)
                        {
                            //we are looking at the point at (rowOutput,colOutput) in the augmented picture
                            //we will first compute the coordinate (rowInput,colInput) of this pixel in the original (not augmented) picture
                            int rowInput = rowOutput;
                            int colInput = colOutput;

                            if (policyMayChangeCoordinates)
                            {
                                //we compute the original row index (that will be transformed to 'rowOutput' after Data Augmentation)
                                rowInput = UnconvertRow(rowOutput, colOutput, unconvertAX, unconvertBX, unconvertCX);
                                if (rowInput < 0)
                                {
                                    rowInput = fillMode == ImageDataGenerator.FillModeEnum.Reflect ? Math.Abs(rowInput + 1) : 0;
                                }
                                if (rowInput >= nbRows)
                                {
                                    rowInput = fillMode == ImageDataGenerator.FillModeEnum.Reflect ? (nbRows - 1 - (rowInput - nbRows)) : (nbRows - 1);
                                }
                                rowInput = Math.Min(Math.Max(0, rowInput), nbRows - 1);
                                Debug.Assert(rowInput >= 0 && rowInput < nbRows);

                                //we compute the original column index (that will be transformed to 'colOutput' after Data Augmentation)
                                colInput = UnconvertCol(rowOutput, colOutput, unconvertAY, unconvertBY, unconvertCY);
                                if (colInput < 0)
                                {
                                    colInput = fillMode == ImageDataGenerator.FillModeEnum.Reflect ? Math.Abs(colInput + 1) : 0;
                                }
                                if (colInput >= nbCols)
                                {
                                    colInput = fillMode == ImageDataGenerator.FillModeEnum.Reflect ? (nbCols - 1 - (colInput - nbCols)) : (nbCols - 1);
                                }
                                colInput = Math.Min(Math.Max(0, colInput), nbCols - 1);
                                Debug.Assert(colInput >= 0 && colInput < nbCols);
                            }

                            var realPrevious = (i == 0) ? xInputMiniBatch : previous;
                            var augmentedValue = policy.AugmentedValue(indexInMiniBatch, channel, realPrevious, rowInput, colInput, next, rowOutput, colOutput);
                            next[outputPictureIdx + colOutput] = augmentedValue;
                        }
                    }
                }
                var tmp = previous;
                previous = next;
                next = tmp;
            }
            subPolicy.ForEach(x => x.UpdateY(yMiniBatch, indexInMiniBatch, indexInMiniBatchToCategoryIndex));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int UnconvertRow(int row, int col, double unconvertAX, double unconvertBX, double unconvertCX)
        {
            return CoordinateToColumnIndex(unconvertAX * row + unconvertBX * col + unconvertCX);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int UnconvertCol(int row, int col, double unconvertAY, double unconvertBY, double unconvertCY)
        {
            return CoordinateToColumnIndex(unconvertAY * row + unconvertBY * col + unconvertCY);
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int CoordinateToColumnIndex(double coordinate)
        {
            return (int)Math.Floor(coordinate);
        }
    }
}