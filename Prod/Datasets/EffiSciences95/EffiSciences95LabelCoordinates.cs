using System.Drawing;
using System.Globalization;
namespace SharpNet.Datasets.EffiSciences95;


/// <summary>
/// for each picture, this class contains the estimate coordinates of the box around the label ("old" or "young") found in this picture
/// </summary>
public class EffiSciences95LabelCoordinates
{
    public EffiSciences95LabelCoordinates(int no, string label, int col_start, int row_start, int width, int height, double density, string descDensity, double confidenceLevel)
    {
        this.No = no;
        this.Label = label;
        this.Col_start = col_start;
        this.Row_start = row_start;
        this.Width = width;
        this.Height = height;
        this.Density = density;
        this.DescDensity = descDensity;
        this.ConfidenceLevel = confidenceLevel;
    }

    /// <summary>
    /// id of the picture
    /// </summary>
    public readonly int No; 
    /// <summary>
    /// for the training dataset:
    ///     label of the picture ("y" for "young" label , "o" for "old" label)
    /// fot the test dataset:
    ///     empty string
    /// </summary>
    public readonly string Label;
    /// <summary>
    /// coordinates of the start of the label box (from the left)
    /// </summary>
    private readonly int Col_start;
    /// <summary>
    /// coordinates of the start of the label box (from the top)
    /// </summary>
    private readonly int Row_start;
    /// <summary>
    /// with of the label box
    /// </summary>
    public readonly int Width;
    /// <summary>
    /// height of the label box
    /// </summary>
    public readonly int Height;
    /// <summary>
    /// percentage of pixels associated with the text of the label
    /// </summary>
    private readonly double Density;
    /// <summary>
    /// the density (between 0% and 100%) of pixels associated with the text of the label for each part of the box (top-left, top-right, bottom-left, bottom-right)
    /// example: 19_35_46_52
    /// this is used to compute the field ConfidenceLevel below (the probability that the extracted box actually contains a label)
    /// See method LabelFinder.ComputeConfidenceLevel
    /// </summary>
    private readonly string DescDensity;
    /// <summary>
    /// probability that the extracted box actually contains a label ("old" or "young")
    /// See method LabelFinder.ComputeConfidenceLevel for the actual computation
    /// </summary>
    public readonly double ConfidenceLevel;

    public override string ToString()
    {
        return No + ";" + Label + ";" + +Col_start + ";" + Row_start + ";" + Width + ";" + Height + ";" + Density + ";" + DescDensity + ";" + ConfidenceLevel;
    }
    public static EffiSciences95LabelCoordinates ValueOfLine(string line)
    {
        if (string.IsNullOrWhiteSpace(line))
        {
            return null;
        }
        var splitted = line.Split(";");
        var res = new EffiSciences95LabelCoordinates
        (
            no: int.Parse(splitted[0]),
            label: splitted[1],
            col_start: int.Parse(splitted[2]),
            row_start: int.Parse(splitted[3]),
            width: int.Parse(splitted[4]),
            height: int.Parse(splitted[5]),
            density: double.Parse(splitted[6], CultureInfo.InvariantCulture),
            descDensity: splitted[7],
            confidenceLevel: double.Parse(splitted[8], CultureInfo.InvariantCulture)
        );
        return res;
    }
    public Rectangle Shape => new (Col_start, Row_start, Width, Height);
    public bool HasKnownLabel => Label == "y" || Label == "o";
    public bool IsClearlyBetterThan(EffiSciences95LabelCoordinates other)
    {
        if (other == null)
        {
            return true;
        }
        if (IsEmpty)
        {
            return false;
        }
        return other.IsEmpty;
    }
    public static EffiSciences95LabelCoordinates Empty(int pictureId, string Label)
    {
        return new EffiSciences95LabelCoordinates(pictureId, Label, 0, 0, 0, 0, 0, "", 0);
    }
    public bool IsEmpty => Width == 0;
}