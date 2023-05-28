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
            CpuTensor<float> xOriginalMiniBatch,
            CpuTensor<float> xDataAugmentedMiniBatch,
            CpuTensor<float> yDataAugmentedMiniBatch,
            Func<int, int> indexInOriginalMiniBatchToCategoryIndex,
            ImageDataGenerator.FillModeEnum fillMode,
            CpuTensor<float> xBufferForDataAugmentedMiniBatch)
        {
            //we ensure that all tensors shape are 4D 'NCHW' tensors  (where N is the batch size, C is the number of channels, H is the height and W is the width)
            (xOriginalMiniBatch, xDataAugmentedMiniBatch, xBufferForDataAugmentedMiniBatch) = ImageDataGenerator.To_NCHW(xOriginalMiniBatch, xDataAugmentedMiniBatch, xBufferForDataAugmentedMiniBatch);
            
            Debug.Assert(xOriginalMiniBatch.SameShape(xDataAugmentedMiniBatch));
            Debug.Assert(xBufferForDataAugmentedMiniBatch != null);
            Debug.Assert(xDataAugmentedMiniBatch.SameShape(xBufferForDataAugmentedMiniBatch));

            if (subPolicy.Count == 0)
            {
                xOriginalMiniBatch.CopyTo(xDataAugmentedMiniBatch);
                return;
            }
            var nbRows = xOriginalMiniBatch.Shape[2];
            var nbCols = xOriginalMiniBatch.Shape[3];
            var previous = (subPolicy.Count % 2 == 1)? xBufferForDataAugmentedMiniBatch : xDataAugmentedMiniBatch;
            var next = (subPolicy.Count % 2 == 1)? xDataAugmentedMiniBatch: xBufferForDataAugmentedMiniBatch;
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

                for (int channel = 0; channel < xOriginalMiniBatch.Shape[1]; ++channel)
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
                                rowInput = FixColInput(fillMode, nbRows, rowInput);
                                Debug.Assert(rowInput >= 0 && rowInput < nbRows);

                                //we compute the original column index (that will be transformed to 'colOutput' after Data Augmentation)
                                colInput = UnconvertCol(rowOutput, colOutput, unconvertAY, unconvertBY, unconvertCY);
                                colInput = FixColInput(fillMode, nbCols, colInput);
                                Debug.Assert(colInput >= 0 && colInput < nbCols);
                            }

                            var realPrevious = (i == 0) ? xOriginalMiniBatch : previous;
                            var augmentedValue = policy.AugmentedValue(indexInMiniBatch, channel, realPrevious, rowInput, colInput, next, rowOutput, colOutput);
                            //!D try to use same Span
                            next[outputPictureIdx + colOutput] = augmentedValue;
                        }
                    }
                }
                (previous, next) = (next, previous);
            }
            subPolicy.ForEach(x => x.UpdateY(yDataAugmentedMiniBatch, indexInMiniBatch, indexInOriginalMiniBatchToCategoryIndex));
        }

        /// <summary>
        /// ensure that the column index 'colInput' is in the range [0, nbCols-1]
        /// always return a value in range [0, nbCols-1]
        /// </summary>
        /// <param name="fillMode"></param>
        /// <param name="nbCols">the total number of columns</param>
        /// <param name="colInput">the index to fix</param>
        /// <returns></returns>
        private static int FixColInput(ImageDataGenerator.FillModeEnum fillMode, int nbCols, int colInput)
        {
            if (colInput >= 0 && colInput<nbCols)
            {
                //no need to fix 'colInput' : it is already in the range [0,nbCols-1]
                return colInput;
            }

            if (fillMode == ImageDataGenerator.FillModeEnum.Reflect)
            {
                var colInputReflectResult = colInput < 0 ? Math.Abs(colInput + 1) : nbCols - 1 - (colInput - nbCols);
                return Math.Min(Math.Max(0, colInputReflectResult), nbCols - 1);
            }
            if (fillMode == ImageDataGenerator.FillModeEnum.Nearest)
            {
                if (colInput < 0)
                {
                    return 0;
                }
                return nbCols - 1;
            }
            Debug.Assert(fillMode == ImageDataGenerator.FillModeEnum.Modulo);
            return Utils.AlwaysPositiveModulo(colInput, nbCols);
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