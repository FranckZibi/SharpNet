using System;
using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.CPU;
using SharpNet.DataAugmentation.Operations;

namespace SharpNet.DataAugmentation
{
    public static class SubPolicy
    {
        public static void Apply(List<Operation> subPolicy,
            int indexInMiniBatch,
            CpuTensor<float> xOriginalMiniBatch,
            CpuTensor<float> xDataAugmentedMiniBatch,
            CpuTensor<float> yMiniBatch,
            Func<int, int> indexInMiniBatchToCategoryId,
            ImageDataGenerator.FillModeEnum fillMode)
        {
            var unconvertCX = Unconvert_Slow(subPolicy, 0, 0).row;
            var unconvertAX = (Unconvert_Slow(subPolicy, 1, 0).row - unconvertCX);
            var unconvertBX = (Unconvert_Slow(subPolicy, 0, 1).row - unconvertCX);
            unconvertCX += 1e-8;

            var unconvertCY = Unconvert_Slow(subPolicy, 0, 0).col;
            var unconvertAY = (Unconvert_Slow(subPolicy, 1, 0).col - unconvertCY);
            var unconvertBY = (Unconvert_Slow(subPolicy, 0, 1).col - unconvertCY);
            unconvertCY += 1e-8;

            var nbRows = xDataAugmentedMiniBatch.Shape[2];
            var nbCols = xDataAugmentedMiniBatch.Shape[3];

            for (int channel = 0; channel < xDataAugmentedMiniBatch.Shape[1]; ++channel)
            {
                for (int rowOutput = 0; rowOutput < nbRows; ++rowOutput)
                {
                    var outputPictureIdx = xDataAugmentedMiniBatch.Idx(indexInMiniBatch, channel, rowOutput, 0);
                    for (int colOutput = 0; colOutput < nbCols; ++colOutput)
                    {
                        //we are looking at the point at (rowOutput,colOutput) in the augmented picture
                        //we will first compute the coordinate (rowInput,colInput) of this pixel in the original (not augmented) picture

                        //we compute the original row index (that will be transformed to 'rowOutput' after Data Augmentation)
                        var rowInput = UnconvertRow(rowOutput, colOutput, unconvertAX, unconvertBX, unconvertCX);
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
                        var colInput = UnconvertCol(rowOutput, colOutput, unconvertAY, unconvertBY, unconvertCY);
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

                        var valueInOriginalPicture = xOriginalMiniBatch.Get(indexInMiniBatch, channel, rowInput, colInput);
                        var augmentedValue = AugmentedValue(subPolicy, valueInOriginalPicture, indexInMiniBatch, xOriginalMiniBatch, xDataAugmentedMiniBatch, channel, rowOutput, colOutput);
                        xDataAugmentedMiniBatch[outputPictureIdx + colOutput] = augmentedValue;
                    }
                }
            }
            subPolicy.ForEach(x => x.UpdateY(yMiniBatch, indexInMiniBatch, indexInMiniBatchToCategoryId));
        }

        private static float AugmentedValue(IEnumerable<Operation> operations, float originalValue,
            int indexInMiniBatch, CpuTensor<float> xOriginalMiniBatch, CpuTensor<float> xDataAugmentedMiniBatch,
            int channelOutput, int rowOutput, int colOutput)
        {
            var augmentedValue = originalValue;
            foreach (var o in operations)
            {
                augmentedValue = o.AugmentedValue(augmentedValue, indexInMiniBatch, xOriginalMiniBatch, xDataAugmentedMiniBatch, channelOutput, rowOutput, colOutput);
            }
            return augmentedValue;
        }

        private static int UnconvertRow(int row, int col, double unconvertAX, double unconvertBX, double unconvertCX)
        {
            return CoordinateToColumnIndex(unconvertAX * row + unconvertBX * col + unconvertCX);
        }

        private static int UnconvertCol(int row, int col, double unconvertAY, double unconvertBY, double unconvertCY)
        {
            return CoordinateToColumnIndex(unconvertAY * row + unconvertBY * col + unconvertCY);
        }
        private static int CoordinateToColumnIndex(double coordinate)
        {
            return (int)Math.Floor(coordinate);
        }

        private static (double row, double col) Unconvert_Slow(IReadOnlyList<Operation> operations, double row, double col)
        {
            (double rowTmp, double colTmp) res = (row, col);
            for (int i = operations.Count - 1; i >= 0; --i)
            {
                res = operations[i].Unconvert_Slow(res.rowTmp, res.colTmp);
            }
            return res;
        }
    }
}