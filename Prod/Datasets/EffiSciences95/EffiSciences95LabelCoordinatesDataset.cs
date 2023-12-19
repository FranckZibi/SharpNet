using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace SharpNet.Datasets.EffiSciences95;

/// <summary>
/// contains the list of label coordinates (class EffiSciences95LabelCoordinates) for a specific dataset (either the labeled one or the Unlabeled one)
/// </summary>
public class EffiSciences95LabelCoordinatesDataset
{
    /// <summary>
    /// the directory where the pictures are stored (either "Labeled" or "Unlabeled")
    /// </summary>
    private readonly string DatasetDirectory;
    private readonly string Header;
    public readonly Dictionary<int, EffiSciences95LabelCoordinates> Content;
    
    public EffiSciences95LabelCoordinatesDataset(string datasetDirectory)
    {
        DatasetDirectory = datasetDirectory;
        Content = new Dictionary<int, EffiSciences95LabelCoordinates>();
        var lines = File.ReadAllLines(Path);
        Header = lines[0];
        for (var i = 1; i < lines.Length; i++)
        {
            var labelCoordinates = EffiSciences95LabelCoordinates.ValueOfLine(lines[i]);
            if (labelCoordinates != null)
            {
                Content[labelCoordinates.No] = labelCoordinates;
            }
        }
    }
    private string Path => System.IO.Path.Combine(EffiSciences95Utils.IDMDirectory, "EffiSciences95_"+DatasetDirectory+ ".csv");

    public void Save()
    {
        File.Copy(Path, Path + "." + DateTime.Now.Ticks , true);
        var sb = new StringBuilder();
        sb.Append(Header);
        foreach (var row in Content.OrderBy(c => c.Key))
        {
            sb.Append(Environment.NewLine+row.Value);
        }
        File.WriteAllText(Path, sb.ToString());
    }
}
