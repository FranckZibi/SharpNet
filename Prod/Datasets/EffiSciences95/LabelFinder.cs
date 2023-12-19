using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using SharpNet.Pictures;
// ReSharper disable ConditionIsAlwaysTrueOrFalse

namespace SharpNet.Datasets.EffiSciences95;


/// <summary>
/// this class is in charge of finding the coordinates of the label in the picture
/// </summary>
public static class LabelFinder
{
    /// <summary>
    /// for each picture in directory 'directory' (either "Labeled" or "Unlabeled"),
    /// this method computes the estimate coordinates of the box containing the label ("old" or "young")
    /// </summary>
    /// <param name="directory">either Labeled or Unlabeled</param>
    public static void FindLabelCoordinates(string directory)
    {
        var cache = new RGBColorFactoryWithCache(false);
        using var datasetSample = new EffiSciences95DatasetSample();
        var dataset = new EffiSciences95LabelCoordinatesDataset(directory);

        int totalProcessed = 0;
        void ProcessFileName(int id)
        {
            var srcPath = Path.Combine(EffiSciences95Utils.DataDirectory, directory, id + ".jpg");
            if (!File.Exists(srcPath))
            {
                return;
            }

            lock (LockObject)
            {
                ++totalProcessed;
                if (totalProcessed % 100 == 0)
                {
                    Console.WriteLine($"Total processed: {totalProcessed}");
                }
            }


            using var bmp = new Bitmap(srcPath);
            var bmpContent = BitmapContent.ValueFomSingleRgbBitmap(bmp);
            var rgbColorContent = bmpContent.RGBColorContent(cache);
            var roots = KMeanRoots;

            var KMeanIndex = MatrixTools.ToKMeanIndex(rgbColorContent, roots, ColorDistances, HyperParam3, 2);
            var bestLabelCoordinate = EffiSciences95LabelCoordinates.Empty(id, "");

            
            for (int i = 0; i < roots.Count; ++i)
            {
                var m = MatrixTools.ExtractBoolMatrix(KMeanIndex, i);
                MatrixTools.RemoveSingleIsolatedElements(m);
                var countMatrix = MatrixTools.CreateCountMatrix(m);
                var allLabelCoordinates = LookForLabelCoordinates(directory, id, i, m, countMatrix, 0, m.GetLength(0) - 1, 0, m.GetLength(1) - 1);

                if (allLabelCoordinates.Count == 0)
                {
                    continue;
                }
                var elementCount = MatrixTools.CountTrue(m);
                if (elementCount <= 100)
                {
                    continue;
                }
                allLabelCoordinates = allLabelCoordinates.OrderByDescending(a => a.ConfidenceLevel).ToList().Take(2).ToList();
                foreach (var labelCoordinate in allLabelCoordinates)
                {
                    if (bestLabelCoordinate.IsEmpty || labelCoordinate.ConfidenceLevel > bestLabelCoordinate.ConfidenceLevel)
                    {
                        bestLabelCoordinate = labelCoordinate;
                    }
                }
            }

            EffiSciences95LabelCoordinates existing;
            lock (dataset)
            {
                if (!dataset.Content.TryGetValue(bestLabelCoordinate.No, out existing))
                {
                    // ReSharper disable once RedundantAssignment
                    existing = null;
                }
            }

            if (bestLabelCoordinate.IsClearlyBetterThan(existing))
            {
                lock (dataset)
                {
                    dataset.Content[bestLabelCoordinate.No] = bestLabelCoordinate;
                }
            }
        }
        Parallel.For(0, 70000, ProcessFileName);

        dataset.Save();
    }


    private static readonly double[] hyperParam1 = { 0.2, 0.25, 0.2, 0.1, 0.1 }; private static readonly double[] HyperParam2 = { 0.7, 0.7, 0.7, 0.55, 0.55}; private static readonly double[] HyperParam3 = { 0.5,0.4,0.5,0.03,0.3}; private static readonly double[] HyperParam4 = { 1.0, 1.0, 1.0, 0.9, 0.9}; private static readonly Func<RGBColor, RGBColor, double>[] ColorDistances = { RGBColor.LabColorDistance, RGBColor.LabColorDistance, RGBColor.YellowLabColorDistance, RGBColor.BlackLabColorDistance, RGBColor.WhiteLabColorDistance}; private static readonly List<RGBColor> KMeanRoots = new() { new(20, 20, 190), new(220, 30, 30), new(255, 255, 0), new(0, 0, 0), new(250, 250, 250) };


    private static readonly object LockObject = new ();

    /// <summary>
    /// compute the probability that the rectangle 'rect' contains a valid label (either old or young)
    /// </summary>
    /// <param name="rect"></param>
    /// <param name="countMatrix"></param>
    /// <param name="i"></param>
    /// <returns></returns>
    private static double ComputeConfidenceLevel(Rectangle rect, int[,] countMatrix, int i)
    {
        var row_start = rect.Top;
        var row_end = rect.Bottom-1;
        var col_start = rect.Left;
        var col_end = rect.Right-1;
        var top_left_percentage = MatrixTools.Density(countMatrix, row_start, row_end - rect.Height / 2, col_start, col_end - rect.Width / 2);
        var top_right_percentage = MatrixTools.Density(countMatrix, row_start, row_end - rect.Height / 2, col_start + rect.Width / 2, col_end);
        var bottom_left_percentage = MatrixTools.Density(countMatrix, row_start + rect.Height / 2, row_end, col_start, col_end - rect.Width / 2);
        var bottom_right_percentage = MatrixTools.Density(countMatrix, row_start + rect.Height / 2, row_end, col_start + rect.Width / 2, col_end);
        double p = HyperParam4[i];
        foreach (var d in new[] {top_left_percentage, top_right_percentage, bottom_left_percentage,bottom_right_percentage})
        {
            p *= Density_to_ValidLabelCoordinatesProbability(d, i);
        }
        p *= LabelCoordinates_to_Probability(rect.Height, rect.Width);
        return p;
    }


    private static readonly List<Tuple<int, int>> OldFormat = new()
    {
        Tuple.Create(16, 31),
        Tuple.Create(17, 30),
        Tuple.Create(17, 38),
        Tuple.Create(18, 34),
        Tuple.Create(18, 36),
        Tuple.Create(19, 38),
        Tuple.Create(20, 43),
        Tuple.Create(22, 46),
        Tuple.Create(23, 43),
        Tuple.Create(24, 49),
        Tuple.Create(25, 28),
        Tuple.Create(26, 50),
        Tuple.Create(26, 51),
        Tuple.Create(27, 34),
        Tuple.Create(28, 31),
        Tuple.Create(28, 32),
        Tuple.Create(29, 53),
        Tuple.Create(30, 36),
        Tuple.Create(32, 37),
        Tuple.Create(35, 43),
        Tuple.Create(35, 43),
        Tuple.Create(36, 45),
        Tuple.Create(39, 48),
        Tuple.Create(42, 48),
    };
    private static readonly List<Tuple<int, int>> YoungFormat = new()
    {
        Tuple.Create(16,67),
        Tuple.Create(17,66),
        Tuple.Create(19,77),
        Tuple.Create(20,71),
        Tuple.Create(20,88),
        Tuple.Create(22,101),
        Tuple.Create(25,105),
        Tuple.Create(26,103),
        Tuple.Create(26,105),
        Tuple.Create(26,112),
        Tuple.Create(26,118),
        Tuple.Create(28,61),
        Tuple.Create(28,63),
        Tuple.Create(30,69),
        Tuple.Create(34,77),
        Tuple.Create(37,79),
        Tuple.Create(40,86),
        Tuple.Create(42,90),
        Tuple.Create(45,96),
        Tuple.Create(46,100),
        Tuple.Create(50,106)
    };



    
    private static double LabelCoordinates_to_Probability(int height, int width)
    {
        double high = 0.0;
        foreach (var d in OldFormat.Union(YoungFormat))
        {
            double current = 1.0;
            current *= Math.Min(d.Item1, height) / Math.Max(d.Item1, (double)height);
            current *= Math.Min(d.Item2, width) / Math.Max(d.Item2, (double)width);
            high = Math.Max(high, current);
        }
        return high;
    }

    
    private static double Density_to_ValidLabelCoordinatesProbability(double density, int i)
    {
        var a = hyperParam1[i];
        var b = HyperParam2[i];
        if (density >= a && density <= b)
        {
            return 1.0;
        }
        if (density < a)
        {
            return Math.Max(0.25, density / a);
        }
        if (density > b)
        {
            return Math.Max(0.25, b / density);
        }
        return 1.0;
    }


    private static bool ValidRowForLabel(int[,] countMatrix, int row, int col_start, int col_end)
    {
        int nonEmptyElements = MatrixTools.RowCount(countMatrix, row, col_start, col_end);
        if (nonEmptyElements < 3)
        {
            return false;
        }
        int first_non_empty_idx = MatrixTools.FirstNonEmptyColInRow(countMatrix, row, col_start, col_end);
        int last_non_empty_col = MatrixTools.LastNonEmptyColInRow(countMatrix, row, col_start, col_end);
        int length_with_non_empty = last_non_empty_col - first_non_empty_idx+1;
        if (length_with_non_empty <= 12)
        {
            return false;
        }
        if (length_with_non_empty >= 30 && nonEmptyElements > 0.9 * length_with_non_empty)
        {
            return false;
        }
        return true;
    }


    private static bool ValidColForLabel(bool[,] m, int[,] countMatrix, int col, int row_start, int row_end)
    {
        int nonEmptyElements = MatrixTools.ColCount(countMatrix, col, row_start, row_end);
        if (nonEmptyElements < 5)
        {
            return false;
        }

        int first_non_empty_idx = MatrixTools.FirstNonEmptyRowInCol(countMatrix, col, row_start, row_end);
        int last_non_empty_idx = MatrixTools.LastNonEmptyRowInCol(countMatrix, col, row_start, row_end);
        int length_with_non_empty = last_non_empty_idx - first_non_empty_idx + 1;
        if (length_with_non_empty <= 5)
        {
            return false;
        }
        if (length_with_non_empty >= 60 && nonEmptyElements > 0.9 * length_with_non_empty)
        {
            return false;
        }

        return true;
    }


    private static List<EffiSciences95LabelCoordinates> LookForLabelCoordinates(string directory, int fileNameIndex, int i, bool[,] m, int[,] countMatrix, int row_start, int row_end, int col_start, int col_end)
    {
        var res = new List<EffiSciences95LabelCoordinates>();
        var validRows = new bool[row_end-row_start+1];
        for (int row = row_start; row <= row_end; ++row)
        {
            validRows[row - row_start] = ValidRowForLabel(countMatrix, row, col_start, col_end);
        }
        MatrixTools.MakeValidIfHasValidWithinDistance(validRows,3);

        var validCols= new bool[col_end - col_start + 1];
        for (int col = col_start; col <= col_end; ++col)
        {
            validCols[col - col_start] = ValidColForLabel(m, countMatrix, col, row_start, row_end);
        }
        MatrixTools.MakeValidIfHasValidWithinDistance(validCols,8);

        for (int row = row_start; row <= row_end; ++row)
        {
            if (!validRows[row - row_start] && MatrixTools.RowCount(countMatrix, row, col_start, col_end) != 0)
            {
                MatrixTools.SetRow(m, row, col_start, col_end, false);
            }
        }
        for (int col = col_start; col <= col_end; ++col)
        {
            if (!validCols[col - col_start] && MatrixTools.ColCount(countMatrix, col, row_start, row_end) != 0)
            {
                MatrixTools.SetCol(m, col, row_start, row_end, false);
            }
        }

        List<Tuple<int, int>> validColsIntervals = MatrixTools.ExtractValidIntervals(validCols, col_start, 20, col => MatrixTools.ColCount(countMatrix, col, row_start, row_end) == 0);
        List<Tuple<int, int>> validRowsIntervals = MatrixTools.ExtractValidIntervals(validRows, row_start, 12, row => MatrixTools.RowCount(countMatrix, row, col_start, col_end) == 0);

        if (validColsIntervals.Count == 0 || validRowsIntervals.Count == 0)
        {
            return res;
        }
        if (validColsIntervals.Count == 1 && validRowsIntervals.Count == 1)
        {
            int row_start0 = validRowsIntervals[0].Item1;
            int row_end0 = validRowsIntervals[0].Item2;
            int col_start0 = validColsIntervals[0].Item1;
            int col_end0 = validColsIntervals[0].Item2;
            while (row_start0 < row_end0 && MatrixTools.RowCount(countMatrix, row_start0, col_start0, col_end0) == 0)
            {
                ++row_start0;
            }
            while (row_end0 > row_start0 && MatrixTools.RowCount(countMatrix, row_end0, col_start0, col_end0) == 0)
            {
                --row_end0;
            }
            while (col_start0 < col_end0 && MatrixTools.ColCount(countMatrix, col_start0, row_start0, row_end0) == 0)
            {
                ++col_start0;
            }
            while (col_end0 > col_start0 && MatrixTools.ColCount(countMatrix, col_end0, row_start0, row_end0) == 0)
            {
                --col_end0;
            }
            int height0 = row_end0 - row_start0 + 1;
            int width0 = col_end0 - col_start0 + 1;
            if (width0 < height0)
            {
                return res;
            }

            var density = MatrixTools.Density(countMatrix, row_start0, row_end0, col_start0, col_end0);
            if (density < hyperParam1[i] - 0.1 || density > HyperParam2[i] + 0.1)
            {
                return res;
            }


            var top_left_percentage = MatrixTools.Density(countMatrix, row_start0, row_end0 - height0 / 2, col_start0, col_end0 - width0 / 2);
            var top_right_percentage = MatrixTools.Density(countMatrix, row_start0, row_end0 - height0 / 2, col_start0 + width0 / 2, col_end0);
            var bottom_left_percentage = MatrixTools.Density(countMatrix, row_start0 + height0 / 2, row_end0, col_start0, col_end0 - width0 / 2);
            var bottom_right_percentage = MatrixTools.Density(countMatrix, row_start0 + height0 / 2, row_end0, col_start0 + width0 / 2, col_end0);
            var densityDesc = (int)(100 * top_left_percentage) + "_" + (int)(100 * top_right_percentage) + "_" + (int)(100 * bottom_left_percentage) + "_" + (int)(100 * bottom_right_percentage);


            var rect = new Rectangle(col_start0, row_start0, width0, height0);
            var row = new EffiSciences95LabelCoordinates(fileNameIndex,
                "",
                col_start0, row_start0, width0, height0,
                Math.Round(density,3),
                densityDesc,
                Math.Round(ComputeConfidenceLevel(rect, countMatrix, i),3)
            );
            return new List<EffiSciences95LabelCoordinates> { row };
        }
        foreach (var colInterval in validColsIntervals)
        foreach (var rowInterval in validRowsIntervals)
        {
            res.AddRange(LookForLabelCoordinates(directory, fileNameIndex, i, m, countMatrix, rowInterval.Item1, rowInterval.Item2, colInterval.Item1, colInterval.Item2));
        }
        return res;
    }
}