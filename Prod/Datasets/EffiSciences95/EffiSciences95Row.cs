using System;
using System.Drawing;
using System.Globalization;
using System.Linq;
// ReSharper disable InconsistentNaming

namespace SharpNet.Datasets.EffiSciences95;

public class EffiSciences95Row

{
    public string GuessLabel()
    {
        return ComputeYoungProba() > ComputeOldProba() ? "y?" : "o?";
    }

    public static readonly double[] DefaultHyperParameters = { 2, 1.647, 0.517, 0.717, 2.086, 0.67, 2.613, 0.519, 0.67, 0.602, 0.802};

    public double ComputeYoungProba(double[] hyperParameters = null)
    {
        hyperParameters ??= DefaultHyperParameters;
        var scale = Scale;
        var dens = DescDensity.Split("_").Select(e=> double.Parse(e, CultureInfo.InvariantCulture)).ToArray();
         var (a, b, c, d) = (dens[0],dens[1],dens[2],dens[3]);
         var proba = 1.0;
         if (scale < hyperParameters[0])
         {
             proba *= scale / 2;
         }
         if (scale < hyperParameters[1])
         {
             proba *= hyperParameters[2];
         }
         if (a < c)
         {
             proba *= hyperParameters[3];
         }
         if (b < d)
         {
             proba *= hyperParameters[3];
         }
         return proba;
    }

    public double ComputeOldProba(double[] hyperParameters = null)
    {
        hyperParameters ??= DefaultHyperParameters;
        var scale = Scale;
        var dens = DescDensity.Split("_").Select(e => double.Parse(e, CultureInfo.InvariantCulture)).ToArray();
        var (a, b, c, d) = (dens[0], dens[1], dens[2], dens[3]);
        var proba = 1.0;
        if (scale > hyperParameters[4])
        {
            proba *= hyperParameters[5];
            if (scale > hyperParameters[6])
            {
                proba *= hyperParameters[7];
            }
        }
        if (a > b)
        {
            proba *= hyperParameters[8];
        }
        if (a > c)
        {
            proba *= hyperParameters[9];
        }
        if (b > d)
        {
            proba *= hyperParameters[10];
        }

        return proba;
    }


    public static EffiSciences95Row Empty(int No, string Label)
    {
        return new EffiSciences95Row(No, Label, 0, 0, 0, 0, 0, 0, "", "", 0, "");
    }

    public bool IsEmpty => Width == 0;
    public EffiSciences95Row(int no, string label, int col_start, int row_start, int width, int height,
        double scale, double density, string descDensity, string freeText, double confidenceLevel, string document, string date = null)
    {
        this.No = no;
        this.Label = label;
        this.Col_start = col_start;
        this.Row_start = row_start;
        this.Width = width;
        this.Height = height;
        this.Scale = scale;
        this.Density = density;
        this.DescDensity = descDensity;
        this.FreeText = freeText;
        this.ConfidenceLevel = confidenceLevel;
        this.Document = document;
        this.Date = date ?? DateTime.Now.ToString(EffiSciences95BoxesDataset.dateTimeFormat);
    }

    public readonly int No;
    public string Label;
    private readonly int Col_start;
    private readonly int Row_start;
    public readonly int Width;
    public readonly int Height;
    private readonly double Scale;
    private readonly double Density;
    private readonly string DescDensity;
    private readonly string FreeText;
    public readonly double ConfidenceLevel;
    public readonly string Document;
    private readonly string Date;

    public override string ToString()
    {
        return No + ";" + Label + ";" + +Col_start + ";" + Row_start + ";" + Width + ";" + Height + ";" + Scale + ";" + Density + ";" + DescDensity + ";" + FreeText + ";" + ConfidenceLevel + ";" + Document + ";" + Date;
    }

    public static EffiSciences95Row ValueOfLine(string line)
    {
        if (string.IsNullOrWhiteSpace(line))
        {
            return null;
        }
        var splitted = line.Split(";");
        if (splitted.Length != 13)
        {
            throw new Exception($"invalid line {line}");
        }
        var res = new EffiSciences95Row
        (
            no: int.Parse(splitted[0]),
            label: splitted[1],
            col_start: int.Parse(splitted[2]),
            row_start: int.Parse(splitted[3]),
            width: int.Parse(splitted[4]),
            height: int.Parse(splitted[5]),
            scale: double.Parse(splitted[6], CultureInfo.InvariantCulture),
            density: double.Parse(splitted[7], CultureInfo.InvariantCulture),
            descDensity: splitted[8],
            freeText: splitted[9],
            confidenceLevel: double.Parse(splitted[10], CultureInfo.InvariantCulture),
            document: splitted[11],
            date: splitted[12]
        );
        return res;
    }

    public Rectangle Shape => new (Col_start, Row_start, Width, Height);

    public bool HasBeenValidated => Label == "y" || Label == "o";
    private bool HasBeenDiscarded => Label == "e";


    public bool IsClearlyBetterThan(EffiSciences95Row other)
    {
        if (other == null)
        {
            return true;
        }
        if (IsEmpty)
        {
            return false;
        }
        if (other.IsEmpty)
        {
            return true;
        }
        if (other.HasBeenValidated)
        {
            return false;
        }
        if (HasBeenValidated)
        {
            return true;
        }
        if (other.HasBeenDiscarded && !HasBeenDiscarded && IntersectionOverUnion(Shape, other.Shape) < 0.2)
        {
            return true;
        }
        return false;
    }

    private static double IntersectionOverUnion(Rectangle a, Rectangle b)
    {
        if (a.IntersectsWith(b))
        {
            var intersection = Rectangle.Intersect(a, b);
            var union = Rectangle.Union(a, b);
            return (double)intersection.Width * intersection.Height / (union.Width * union.Height);
        }
        return 0;
    }

}