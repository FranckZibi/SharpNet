using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Threading;

namespace SharpNet.Datasets.EffiSciences95;

public class EffiSciences95BoxesDataset
{
    private readonly bool _isLabeled;
    private readonly string Header;
    public readonly Dictionary<int, EffiSciences95Row> Content;
    public const string dateTimeFormat = "yyyy/MM/dd HH:mm:ss";
    private const string LabeledDocumentDirectory = "C:/Projects/Challenges/EffiSciences95/Data/Labeled/Boxes";
    private const string UnlabeledDocumentDirectory = "C:/Projects/Challenges/EffiSciences95/Data/Unlabeled/Boxes";
    

    public EffiSciences95BoxesDataset(bool isLabeled)
    {
        _isLabeled = isLabeled;
        Content = new Dictionary<int, EffiSciences95Row>();
        var lines = File.ReadAllLines(Path);
        Header = lines[0];
        for (var i = 1; i < lines.Length; i++)
        {
            var row = EffiSciences95Row.ValueOfLine(lines[i]);
            if (row != null)
            {
                Content[row.No] = row;
            }
        }
    }

    public string DocumentDirectory => GetDocumentDirectory(_isLabeled);
    
    public static string GetDocumentDirectory(bool isLabeled) => isLabeled ? LabeledDocumentDirectory : UnlabeledDocumentDirectory;


    public string Path => _isLabeled ? EffiSciences95Utils.LabeledPath : EffiSciences95Utils.UnlabeledPath;


    /// <summary>
    /// perform a HPO on the algo to distinguish between young and old caption
    /// </summary>
    public void YoundOldAccuracyHpo()
    {
        var bestAccuracy = ComputeYoungOrOldAccuracy();
        var bestHpo = (double[])EffiSciences95Row.default_hpo.Clone();
        Console.WriteLine($"Young/Old Accuracy : {bestAccuracy}");

        void Process()
        {
            var r = new Random(Thread.CurrentThread.ManagedThreadId);
            var current = (double[])EffiSciences95Row.default_hpo.Clone();
            for (int i = 0; i < 10000; ++i)
            {
                for (int j = 0; j < current.Length; ++j)
                {
                    current[j] = bestHpo[j];
                    if (r.NextDouble() > 0.8)
                    {
                        current[j] *= (0.9 + 0.2 * r.NextDouble());
                    }
                }

                var accuracy = ComputeYoungOrOldAccuracy(current);
                if (accuracy > bestAccuracy)
                {
                    lock (bestHpo)
                    {
                        bestAccuracy = accuracy;
                        bestHpo = (double[])current.Clone();
                    }
                    Console.WriteLine($"Young/Old new best accuracy : {bestAccuracy} / hpo : {string.Join(",", bestHpo)}");
                }
            }
        }
        Parallel.For(0, 1000, i => Process());
    }

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

    public double ComputeYoungOrOldAccuracy(double[] hpo = null)
    {
        int total = 0;
        int ok = 0;
        foreach (var v in Content.Values.Where(e=>e.HasBeenValidated))
        {
            ++total;
            bool predicted_old = v.ComputeOldProba(hpo)>v.ComputeYoungProba(hpo);
            bool actual_old = v.Label == "o";
            if (actual_old == predicted_old)
            {
                ++ok;
            }
        }
        //Console.WriteLine($"total = {total}, ok = {ok}");
        return ok / (double)Math.Max(total,1);
    }
}