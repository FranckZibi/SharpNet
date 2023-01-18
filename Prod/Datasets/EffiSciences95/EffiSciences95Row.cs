using System;
using System.Drawing;
using System.Globalization;
using System.Linq;

namespace SharpNet.Datasets.EffiSciences95;

public class EffiSciences95Row

{
    public string GuessLabel()
    {
        return ComputeYoungProba() > ComputeOldProba() ? "y?" : "o?";
    }

    //public static readonly double[] default_hpo = new [] { 2, 1.6, 0.5, 0.67, 2.1, 0.67, 2.25, 0.5, 0.67, 0.67, 0.67, };
    public static readonly double[] default_hpo = new[] { 2, 1.647, 0.517, 0.717, 2.086, 0.67, 2.613, 0.519, 0.67, 0.602, 0.802};

    public double ComputeYoungProba(double[] hpo = null)
    {
        hpo = hpo ?? default_hpo;
        var scale = Scale;
        var dens = DescDensity.Split("_").Select(e=> double.Parse(e, CultureInfo.InvariantCulture)).ToArray();
         var (a, b, c, d) = (dens[0],dens[1],dens[2],dens[3]);
         var proba = 1.0;
         if (scale < hpo[0])
         {
             proba *= scale / 2;
         }
         if (scale < hpo[1])
         {
             proba *= hpo[2];
         }
         if (a < c)
         {
             proba *= hpo[3];
         }
         if (b < d)
         {
             proba *= hpo[3];
         }
         return proba;
    }

    public double ComputeOldProba(double[] hpo = null)
    {
        hpo = hpo ?? default_hpo;
        var scale = Scale;
        var dens = DescDensity.Split("_").Select(e => double.Parse(e, CultureInfo.InvariantCulture)).ToArray();
        var (a, b, c, d) = (dens[0], dens[1], dens[2], dens[3]);
        var proba = 1.0;
        if (scale > hpo[4])
        {
            proba *= hpo[5];
            if (scale > hpo[6])
            {
                proba *= hpo[7];
            }
        }
        if (a > b)
        {
            proba *= hpo[8];
        }
        if (a > c)
        {
            proba *= hpo[9];
        }
        if (b > d)
        {
            proba *= hpo[10];
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

    public string FileSuffix
    {
        get
        {
            var rect = Shape;
            return rect.Height + "x" + rect.Width
                   + "_" + (100 * ConfidenceLevel)
                   + "_" + DescDensity;
        }
    }

    public int No;
    public string Label;
    public int Col_start;
    public int Row_start;
    public int Width;
    public int Height;
    public double Scale;
    public double Density;
    public string DescDensity;
    public string FreeText;
    public double ConfidenceLevel;
    public string Document;
    public string Date;

    public override string ToString()
    {
        return No + ";" + Label + ";" + +Col_start + ";" + Row_start + ";" + Width + ";" + Height 
               + ";" + Scale + ";" + Density + ";" + DescDensity + ";" + FreeText + ";" + ConfidenceLevel + ";" + Document + ";" + Date;
    }

    public static EffiSciences95Row ValueOfLine(string line)
    {
        if (string.IsNullOrWhiteSpace(line))
        {
            return null;
        }
        var splitted = line.Split(";");
        if (splitted.Count() != 13)
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
    public bool HasBeenDiscarded => Label == "e";


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