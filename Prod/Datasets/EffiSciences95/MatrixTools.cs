using System;
using System.Collections.Generic;
using System.Linq;
using SharpNet.Pictures;

namespace SharpNet.Datasets.EffiSciences95;


/// <summary>
/// different tools used to make fast computation on a matrix (2D array)
/// it is mainly based on precomputed sub matrix sum (see https://www.techiedelight.com/calculate-sum-elements-sub-matrix-constant-time/)
/// </summary>
public static class MatrixTools
{
    public static int[,] CreateCountMatrix(bool[,] m)
    {
        var countMatrix = new int[m.GetLength(0), m.GetLength(1)];
        for (int row = 0; row < m.GetLength(0); ++row)
        {
            for (int col = 0; col < m.GetLength(1); ++col)
            {
                countMatrix[row, col] =
                    (m[row, col] ? 1 : 0)
                    + Default(countMatrix, row, col - 1, 0)
                    + Default(countMatrix, row - 1, col, 0)
                    - Default(countMatrix, row - 1, col - 1, 0);
            }
        }
        return countMatrix;
    }
    public static List<Tuple<int, int>> ExtractValidIntervals(bool[] validCols, int first_idx, int minLengthOfInterval, Func<int, bool> isEmpty)
    {
        List<Tuple<int, int>> res = new();
        int start = -1;
        for (int row = first_idx; row < first_idx + validCols.Length; ++row)
        {
            int idx_in_validCols = row - first_idx;
            if (validCols[idx_in_validCols])
            {
                if (start == -1)
                {
                    if (isEmpty(row))
                    {
                        continue;
                    }
                    start = row;
                }
                if (idx_in_validCols == validCols.Length - 1)
                {
                    if ((row - start + 1) >= minLengthOfInterval)
                    {
                        res.Add(new Tuple<int, int>(start, row));
                    }
                    start = -1;
                }
            }
            else
            {
                if (start != -1)
                {
                    int end = row - 1;
                    while (isEmpty(end) && end > start)
                    {
                        --end;
                    }
                    if ((end - start + 1) >= minLengthOfInterval)
                    {
                        res.Add(new Tuple<int, int>(start, end));
                    }
                    start = -1;
                }
            }
        }
        return res;
    }
    public static void SetRow(bool[,] m, int row, int col_start, int col_end, bool newValue)
    {
        for (int col = col_start; col <= col_end; ++col)
        {
            m[row, col] = newValue;
        }
    }
    public static void SetCol(bool[,] m, int col, int row_start, int row_end, bool newValue)
    {
        for (int row = row_start; row <= row_end; ++row)
        {
            m[row, col] = newValue;
        }
    }
    public static int[,] ToKMeanIndex(RGBColor[,] allColors, IList<RGBColor> KMeanRoots, Func<RGBColor, RGBColor, double>[] ColorDistances, double[] computeMainColorFromPointsWithinDistances, int remainingTries)
    {
        int[] counts = new int[KMeanRoots.Count];
        var result = new int[allColors.GetLength(0), allColors.GetLength(1)];
        for (int row = 0; row < allColors.GetLength(0); ++row)
            for (int col = 0; col < allColors.GetLength(1); ++col)
            {
                int i = -1;
                result[row, col] = -1;
                double minDistance = 2 * computeMainColorFromPointsWithinDistances.Max();
                for (int j = 0; j < KMeanRoots.Count; ++j)
                {
                    double distance = ColorDistances[j](KMeanRoots[j], allColors[row, col]);
                    if (distance <= minDistance)
                    {
                        minDistance = distance;
                        i = j;
                    }
                }
                if (i >= 0 && minDistance <= computeMainColorFromPointsWithinDistances[i])
                {
                    result[row, col] = i;
                    ++counts[i];
                }
            }

        var to_check_for_count = new int[] { 0, 1, 2, 4};
        if (remainingTries > 0 && to_check_for_count.Any(i => counts[i] >= 1700))
        {

            computeMainColorFromPointsWithinDistances = (double[])computeMainColorFromPointsWithinDistances.Clone();
            foreach (var j in to_check_for_count)
            {
                if (counts[j] > 1700)
                {
                    computeMainColorFromPointsWithinDistances[j] /= 1.5;
                }
            }
            return ToKMeanIndex(allColors, KMeanRoots, ColorDistances, computeMainColorFromPointsWithinDistances, remainingTries - 1);
        }

        return result;
    }
    public static int RowCount(int[,] countMatrix, int row, int col_start, int col_end)
    {
        return Count(countMatrix, row, row, col_start, col_end);
    }
    public static int ColCount(int[,] countMatrix, int col, int row_start, int row_end)
    {
        return Count(countMatrix, row_start, row_end, col, col);
    }
    public static double Density(int[,] countMatrix, int row_start, int row_end, int col_start, int col_end)
    {
        int count = Count(countMatrix, row_start, row_end, col_start, col_end);
        return count / ((row_end - row_start + 1.0) * (col_end - col_start + 1.0));
    }
    public static bool[,] ExtractBoolMatrix(int[,] kmeanTextColorIndex, int index)
    {
        var res = new bool[kmeanTextColorIndex.GetLength(0), kmeanTextColorIndex.GetLength(1)];
        for (int row = 0; row < kmeanTextColorIndex.GetLength(0); ++row)
        {
            for (int col = 0; col < kmeanTextColorIndex.GetLength(1); ++col)
            {
                res[row, col] = kmeanTextColorIndex[row, col] == index;
            }
        }
        return res;
    }
    public static int RemoveSingleIsolatedElements(bool[,] m)
    {
        int removedElements = 0;
        for (int row = 0; row < m.GetLength(0); ++row)
        {
            for (int col = 0; col < m.GetLength(1); ++col)
            {
                if (!m[row, col])
                {
                    continue;
                }
                if (Default(m, row - 1, col - 1, false) || Default(m, row, col - 1, false) || Default(m, row + 1, col - 1, false)
                    || Default(m, row - 1, col + 1, false) || Default(m, row, col + 1, false) || Default(m, row + 1, col + 1, false)
                    || Default(m, row - 1, col, false) || Default(m, row + 1, col, false)
                   )
                {
                    continue;
                }

                ++removedElements;
                m[row, col] = false;
            }
        }
        return removedElements;
    }
    public static int FirstNonEmptyColInRow(int[,] countMatrix, int row, int col_start, int col_end)
    {
        int totalNonEmpty = RowCount(countMatrix, row, col_start, col_end);
        if (totalNonEmpty == 0)
        {
            return -1; // all cells are empty
        }
        var countAtStart = RowCount(countMatrix, row, col_start, col_start);
        if (countAtStart == 1)
        {
            return col_start;
        }
        var lastBeforeNonEmpty = MaximumValidIndex(col_start, col_end, c => RowCount(countMatrix, row, col_start, c) == 0);
        return lastBeforeNonEmpty + 1;
    }
    public static int LastNonEmptyColInRow(int[,] countMatrix, int row, int col_start, int col_end)
    {
        int totalNonEmpty = RowCount(countMatrix, row, col_start, col_end);
        if (totalNonEmpty == 0)
        {
            return -1; // all cells are empty
        }
        int countAtEnd = RowCount(countMatrix, row, col_end, col_end);
        if (countAtEnd == 1)
        {
            return col_end;
        }
        var lastAfterNonEmpty = -MaximumValidIndex(-col_end, -col_start, neg_start => RowCount(countMatrix, row, -neg_start, col_end) == 0);
        return lastAfterNonEmpty - 1;
    }
    public static int FirstNonEmptyRowInCol(int[,] countMatrix, int col, int row_start, int row_end)
    {
        int totalNonEmpty = ColCount(countMatrix, col, row_start, row_end);
        if (totalNonEmpty == 0)
        {
            return -1; // all cells are empty
        }
        int countAtStart = ColCount(countMatrix, col, row_start, row_start);
        if (countAtStart == 1)
        {
            return row_start;
        }
        var lastBeforeNonEmpty = MaximumValidIndex(row_start, row_end, c => ColCount(countMatrix, col, row_start, c) == 0);
        return lastBeforeNonEmpty + 1;
    }
    public static int LastNonEmptyRowInCol(int[,] countMatrix, int col, int row_start, int row_end)
    {
        int totalNonEmpty = ColCount(countMatrix, col, row_start, row_end);
        if (totalNonEmpty == 0)
        {
            return -1; // all cells are empty
        }
        int countAtEnd = ColCount(countMatrix, col, row_end, row_end);
        if (countAtEnd == 1)
        {
            return row_end;
        }
        var lastAfterNonEmpty = -MaximumValidIndex(-row_end, -row_start, neg_start => ColCount(countMatrix, col, -neg_start, row_end) == 0);
        return lastAfterNonEmpty - 1;
    }
    public static void MakeValidIfHasValidWithinDistance(bool[] isValid, int distanceToValidToMakeItValid)
    {
        var subSum = CreateSubSum(isValid);

        int last = isValid.Length - 1;

        while (last > 0 && !isValid[last])
        {
            --last;
        }


        for (int i = 0; i <= last; ++i)
        {
            if (isValid[i])
            {
                continue;
            }
            int countValidWithinDistance = GetSubSum(subSum, i - distanceToValidToMakeItValid, i + distanceToValidToMakeItValid);
            if (countValidWithinDistance != 0)
            {
                isValid[i] = true;
            }
        }

    }
    public static int CountTrue(bool[,] m)
    {
        int res = 0;
        for (int row = 0; row < m.GetLength(0); ++row)
        {
            for (int col = 0; col < m.GetLength(1); ++col)
            {
                if (m[row, col])
                {
                    ++res;
                }
            }
        }
        return res;
    }


    /// <summary>
    /// Find the max index for which IsValid is true using dichotomy search
    /// hypothesis: IsValid[min] is true and will always be true for an interval [min, y] then always false after y
    /// Complexity:         o( log(N) ) 
    /// </summary>
    /// <param name="minLength"></param>
    /// <param name="maxLength"></param>
    /// <param name="isValid"></param>
    /// <returns></returns>
    private static int MaximumValidIndex(int minLength, int maxLength, Func<int, bool> isValid)
    {
        while (minLength < maxLength)
        {
            var middle = (minLength + maxLength + 1) / 2;
            if (isValid(middle))
            {
                minLength = middle;
            }
            else
            {
                maxLength = middle - 1;
            }
        }
        return minLength;
    }
    private static int[] CreateSubSum(bool[] c)
    {
        var res = new int[c.Length];
        for (int i = 0; i < c.Length; ++i)
        {
            res[i] = c[i] ? 1 : 0;
            if (i != 0)
            {
                res[i] += res[i - 1];
            }
        }
        return res;
    }
    private static int GetSubSum(int[] subSum, int start, int end)
    {
        if (start > end || start >= subSum.Length)
        {
            return 0;
        }

        int prevSum = start <= 0 ? 0 : subSum[start - 1];
        int totalSum = end >= subSum.Length ? subSum[^1] : subSum[end];
        return totalSum - prevSum;
    }
    private static int Count(int[,] countMatrix, int row_start, int row_end, int col_start, int col_end)
    {
        int rows = countMatrix.GetLength(0);
        int cols = countMatrix.GetLength(1);
        if (row_start > row_end || col_start > col_end || row_start >= rows || col_start >= cols)
        {
            return 0;
        }
        return countMatrix[row_end, col_end]
               - Default(countMatrix, row_start - 1, col_end, 0)
               - Default(countMatrix, row_end, col_start - 1, 0)
               + Default(countMatrix, row_start - 1, col_start - 1, 0);
    }
    private static bool IsValidCoordinate<T>(T[,] data, int row, int col)
    {
        return row >= 0 && col >= 0 && row < data.GetLength(0) && col < data.GetLength(1);
    }
    private static T Default<T>(T[,] data, int row, int col, T defaultIfInvalid)
    {
        return IsValidCoordinate(data, row, col) ? data[row, col] : defaultIfInvalid;
    }

}