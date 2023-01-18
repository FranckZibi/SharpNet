using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using SharpNet.Pictures;

namespace SharpNet.Datasets.EffiSciences95;

public static class BoxFinder
{
    public static void FindBox(bool isLabeled)
    {
        var cache = new RGBColorFactoryWithCache(false);
        bool debug = false;
        var dataset = new EffiSciences95BoxesDataset(isLabeled);

        int totalProcessed = 0;
        void ProcessFileName(int fileNameIndex)
        {
            //var srcPath = Path.Combine(DataDirectory, "work", fileNameIndex+".jpg");
            var srcPath = Path.Combine(EffiSciences95Utils.DataDirectory, isLabeled?"Labeled":"Unlabeled", fileNameIndex + ".jpg");
            if (!File.Exists(srcPath))
            {
                return;
            }

            lock (lockObject)
            {
                ++totalProcessed;
                if (totalProcessed % 100 == 0)
                {
                    Console.WriteLine($"Total processed: {totalProcessed}");
                }
            }


            using var bmp = new Bitmap(srcPath);
            var bmpContent = BitmapContent.ValueFomSingleRgbBitmap(bmp);
            //bmpContent.Save(targetPath);

            var rgbColorContent = bmpContent.RGBColorContent(cache);
            IList<RGBColor> kmeanColors = initialRootColors;

            var kmeanTextColorIndex = MatrixTools.ToKMeanTextColorIndex(rgbColorContent, kmeanColors, ColorDistances, computeMainColorFromPointsWithinDistances, 2);
            var bestBoxe = EffiSciences95Row.Empty(fileNameIndex, "");

            
            for (int textColorIndex = 0; textColorIndex < kmeanColors.Count; ++textColorIndex)
            {
                var m = MatrixTools.ExtractBoolMatrix(kmeanTextColorIndex, textColorIndex);

                (byte R, byte G, byte B) get(int row, int col, byte r, byte g, byte b)
                {
                    if (m[row, col])
                    {
                        return (r, g, b);
                    }
                    return (161, 169, 164); //grey
                }

                MatrixTools.RemoveSingleIsolatedElements(m);
                var countMatrix = MatrixTools.CreateCountMatrix(m);
                var allBoxes = LookForDigits(isLabeled, fileNameIndex, textColorIndex, m, countMatrix, 0, m.GetLength(0) - 1, 0, m.GetLength(1) - 1);

                if (allBoxes.Count == 0 && !debug)
                {
                    continue;
                }
                var elementCount = MatrixTools.CountTrue(m);
                if (elementCount <= 100)
                {
                    continue;
                }

                List<EffiSciences95Row> boxesToDrawRectangles = new();

                allBoxes = allBoxes.OrderByDescending(a => a.ConfidenceLevel).ToList().Take(2).ToList();

                foreach (var box in allBoxes)
                {
                    if (bestBoxe.IsEmpty || box.ConfidenceLevel > bestBoxe.ConfidenceLevel)
                    {
                        bestBoxe = box;
                        boxesToDrawRectangles.Add(box);
                    }
                }

                if (debug && boxesToDrawRectangles.Count != 0)
                {
                    using Bitmap bmpFamily = bmpContent.AsBitmap(get);
                    using Graphics graphics = Graphics.FromImage(bmpFamily);
                    using Pen pen = new Pen(Color.FromKnownColor(KnownColor.Blue), 2);
                    string rectangleDesc = "";
                    foreach (var row in boxesToDrawRectangles)
                    {
                        rectangleDesc += "_" + row.FileSuffix;
                        graphics.DrawRectangle(pen, row.Shape.Left, row.Shape.Top, row.Shape.Width, row.Shape.Height);
                    }
                    var path = Utils.UpdateFilePathWithPrefixSuffix(srcPath, "", "_" + DateTime.Now.Ticks + "_f" + textColorIndex + "_c_" + elementCount + rectangleDesc);
                    PictureTools.SavePng(bmpFamily, path);
                }
            }

            if (debug && !bestBoxe.IsEmpty)
            {
                using Bitmap bmpFamily = bmpContent.AsBitmap();
                using Graphics graphics = Graphics.FromImage(bmpFamily);
                using Pen pen = new Pen(Color.FromKnownColor(KnownColor.Blue), 2);
                var rect = bestBoxe.Shape;
                graphics.DrawRectangle(pen, rect.Left, rect.Top, rect.Width, rect.Height);
                var rectangleDesc = "_" + bestBoxe.FileSuffix;
                var path = Utils.UpdateFilePathWithPrefixSuffix(srcPath, "", "_000_" + DateTime.Now.Ticks + rectangleDesc);
                PictureTools.SavePng(bmpFamily, path);
            }

            EffiSciences95Row existing;
            lock (dataset)
            {
                if (!dataset.Content.TryGetValue(bestBoxe.No, out existing))
                {
                    existing = null;
                }
            }

            if (bestBoxe.IsClearlyBetterThan(existing))
            {
                if (existing != null && Utils.FileExist(existing.Document))
                {
                    Utils.TryDelete(existing.Document);
                }

                lock (dataset)
                {
                    dataset.Content[bestBoxe.No] = bestBoxe;
                }

                if (isLabeled && !bestBoxe.IsEmpty)
                { 
                    //we do not save the box 'old' / 'young' for the unlabeled dataset
                    using var bmpFamily2 = bmpContent.AsBitmap();
                    using var bmpFamily3 = bmpFamily2.Clone(bestBoxe.Shape, bmpFamily2.PixelFormat);
                    PictureTools.SavePng(bmpFamily3, bestBoxe.Document);
                }
            }
        }
        Parallel.For(0, 70000, ProcessFileName);

        dataset.Save();
    }



    private static readonly double[] min_family_density = new[] { 0.2 /*blue*/, 0.25 /*red*/, 0.2 /*yellow*/, 0.1 /*black*/, 0.1 /*white*/};
    private static readonly double[] max_family_density = new[] { 0.7 /*blue*/, 0.7 /*red*/, 0.7 /*yellow*/, 0.55 /*black*/, 0.55 /*white*/};
    private static readonly double[] computeMainColorFromPointsWithinDistances = new[]{ 0.5,  /* blue */ 0.4, /* red */ 0.5,  /* yellow */ 0.03 /* black */,   0.3 /*  white */ }; private static readonly double[] family_confidence_level = new[] { 1.0 /*blue*/, 1.0 /*red*/, 1.0 /*yellow*/, 0.9 /*black*/, 0.9 /*white*/};
    private static readonly string[] family_names = new[] { "blue", "red", "yellow", "black", "white" };
    private static readonly List<RGBColor> initialRootColors = new List<RGBColor>
    {
        new RGBColor(20, 20 /*10*/, 190 /*200*/ /*210*/),//new RGBColor(0, 0, 255), //blue
        new RGBColor(220, 30, 30 /*20*/ /*15*/),//new RGBColor(220, 20, 20), //red
        new RGBColor(255, 255, 0 /*60*/),//new RGBColor(250, 250, 40), //yellow
        new RGBColor(0, 0, 0), //black
        new RGBColor(250, 250, 250),//new RGBColor(255, 255, 255), //white
    };


    private static readonly Func<RGBColor, RGBColor, double>[] ColorDistances = new[]
    {
        RGBColor.LabColorDistance, // blue
        RGBColor.LabColorDistance, // red
        RGBColor.YellowLabColorDistance, // yellow
        RGBColor.BlackLabColorDistance, // black
        RGBColor.WhiteLabColorDistance, // white
    };



    private static readonly object lockObject = new object();
    private static double ComputeConfidenceLevel(Rectangle rect, int[,] countMatrix, int textColorIndex)
    {
        var row_start = rect.Top;
        var row_end = rect.Bottom-1;
        var col_start = rect.Left;
        var col_end = rect.Right-1;
        var top_left_percentage = MatrixTools.Density(countMatrix, row_start, row_end - rect.Height / 2, col_start, col_end - rect.Width / 2);
        var top_right_percentage = MatrixTools.Density(countMatrix, row_start, row_end - rect.Height / 2, col_start + rect.Width / 2, col_end);
        var bottom_left_percentage = MatrixTools.Density(countMatrix, row_start + rect.Height / 2, row_end, col_start, col_end - rect.Width / 2);
        var bottom_right_percentage = MatrixTools.Density(countMatrix, row_start + rect.Height / 2, row_end, col_start + rect.Width / 2, col_end);
        double proba = family_confidence_level[textColorIndex];
        foreach (var d in new[] {top_left_percentage, top_right_percentage, bottom_left_percentage,bottom_right_percentage})
        {
            proba *= DensityToProba(d, textColorIndex);
        }
        proba *= FormatToProba(rect.Height, rect.Width);
        return proba;
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



    
    private static double FormatToProba(int height, int width)
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
    private static double DensityToProba(double d, int textColorIndex)
    {
        var min = min_family_density[textColorIndex];
        var max = max_family_density[textColorIndex];
        if (d >= min && d <= max)
        {
            return 1.0;
        }
        if (d < min)
        {
            return Math.Max(0.25, d / min);
        }
        if (d > max)
        {
            return Math.Max(0.25, max / d);
        }
        return 1.0;
    }


    private static bool ValidRowForDigits(bool[,] m, int[,] countMatrix, int row, int col_start, int col_end)
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


    private static bool ValidColForDigits(bool[,] m, int[,] countMatrix, int col, int row_start, int row_end)
    {
        int nonEmptyElements = MatrixTools.ColCount(countMatrix, col, row_start, row_end);
        //int rows = row_end - row_start + 1;
        //if (nonEmptyElements < 0.1*rows)
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


    private static List<EffiSciences95Row> LookForDigits(bool isLabeled, int fileNameIndex, int textColorIndex, bool[,] m, int[,] countMatrix, int row_start, int row_end, int col_start, int col_end)
    {
        var res = new List<EffiSciences95Row>();
        //int[,] countMatrix = CreateCountMatrix(m);
        //var rows = m.GetLength(0);
        //var cols = m.GetLength(1);

        //var (a, b) = CountByRowsAndCols(countMatrix, row_start, row_end, col_start, col_end);

        var validRows = new bool[row_end-row_start+1];
        for (int row = row_start; row <= row_end; ++row)
        {
            validRows[row - row_start] = ValidRowForDigits(m, countMatrix, row, col_start, col_end);
        }
        MatrixTools.MakeValidIfHasValidWithinDistance(validRows,3);

        var validCols= new bool[col_end - col_start + 1];
        for (int col = col_start; col <= col_end; ++col)
        {
            validCols[col - col_start] = ValidColForDigits(m, countMatrix, col, row_start, row_end);
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
            if (density < min_family_density[textColorIndex] - 0.1 || density > max_family_density[textColorIndex] + 0.1)
            {
                return res;
            }


            var top_left_percentage = MatrixTools.Density(countMatrix, row_start0, row_end0 - height0 / 2, col_start0, col_end0 - width0 / 2);
            var top_right_percentage = MatrixTools.Density(countMatrix, row_start0, row_end0 - height0 / 2, col_start0 + width0 / 2, col_end0);
            var bottom_left_percentage = MatrixTools.Density(countMatrix, row_start0 + height0 / 2, row_end0, col_start0, col_end0 - width0 / 2);
            var bottom_right_percentage = MatrixTools.Density(countMatrix, row_start0 + height0 / 2, row_end0, col_start0 + width0 / 2, col_end0);
            var densityDesc = (int)(100 * top_left_percentage) + "_" + (int)(100 * top_right_percentage) + "_" + (int)(100 * bottom_left_percentage) + "_" + (int)(100 * bottom_right_percentage);


            var rect = new Rectangle(col_start0, row_start0, width0, height0);
            var row = new EffiSciences95Row(fileNameIndex,
                "",
                col_start0, row_start0, width0, height0,
                Math.Round(width0 / (double)Math.Max(height0,1),3), //scale
                Math.Round(density,3),
                densityDesc,
                family_names[textColorIndex],
                Math.Round(ComputeConfidenceLevel(rect, countMatrix, textColorIndex),3),
                Path.Combine(EffiSciences95BoxesDataset.GetDocumentDirectory(isLabeled), fileNameIndex + ".png")
            );
            row.Label = row.GuessLabel();
            return new List<EffiSciences95Row> { row };
        }
        foreach (var colInterval in validColsIntervals)
        foreach (var rowInterval in validRowsIntervals)
        {
            res.AddRange(LookForDigits(isLabeled, fileNameIndex, textColorIndex, m, countMatrix, rowInterval.Item1, rowInterval.Item2, colInterval.Item1, colInterval.Item2));
        }

        return res;
    }

}