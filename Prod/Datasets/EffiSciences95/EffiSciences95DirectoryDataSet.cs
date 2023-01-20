using System;
using System.Collections.Generic;
using SharpNet.CPU;
using SharpNet.Pictures;

namespace SharpNet.Datasets.EffiSciences95;

public class ThreadSafeRandom
{
    private static readonly Random _global = new Random();
    [ThreadStatic] private static Random _local;

    // ReSharper disable once MemberCanBeMadeStatic.Global
    public int Next(int maxValue)
    {
        if (_local == null)
        {
            int seed;
            lock (_global)
            {
                seed = _global.Next();
            }
            _local = new Random(seed);
        }

        return _local.Next(maxValue);
    }
}

public class EffiSciences95DatasetSample : AbstractDatasetSample
{
    public EffiSciences95DatasetSample() : base(new HashSet<string>())
    {
    }

    public override string[] CategoricalFeatures => new string[0];
    public override string[] IdColumns => new[] { "index" };
    public override string[] TargetLabels => new []{"labels"};
    public override Objective_enum GetObjective()
    {
        return Objective_enum.Classification;
    }

    public override int[] GetInputShapeOfSingleElement()
    {
        return EffiSciences95Utils.Shape_CHW;
    }
    public override int NumClass => 2;
    public override DataSet TestDataset()
    {
        return null;
        //!D return EffiSciences95DirectoryDataSet.ValueOf(false);
    }

    public override DataSet FullTrainingAndValidation()
    {
        return EffiSciences95DirectoryDataSet.ValueOf(true);
    }

    public override EvaluationMetricEnum GetRankingEvaluationMetric()
    {
        return EvaluationMetricEnum.Accuracy;
    }
}

public class EffiSciences95DirectoryDataSet : DirectoryDataSet
{
    private readonly List<EffiSciences95Row> _boxes;
    private readonly ThreadSafeRandom _r = new ();

    public static EffiSciences95DirectoryDataSet ValueOf(bool isLabeled, int maxElementCount = -1)
    {
        var idToBoxes = new EffiSciences95BoxesDataset(isLabeled).Content;
        var idToTextLabel = isLabeled?EffiSciences95Utils.IdToTextTarget(isLabeled):null;
        
        List<List<string>> elementIdToPaths = new();
        List<string> elementIdToDescription = new();
        List<int> elementIdToCategoryIndex = new();
        List<EffiSciences95Row> boxes = new();

        for(int id=0;id<=EffiSciences95Utils.MaxId(isLabeled);++id)
        {
            int textLabel = -1;
            var box = idToBoxes.ContainsKey(id) ? idToBoxes[id] : null;
            if (box!= null && (box.IsEmpty || !box.HasBeenValidated))
            {
                box = null;
            }
            if (isLabeled)
            {
                if (!idToTextLabel.ContainsKey(id))
                {
                    throw new Exception($"no label for {EffiSciences95Utils.IdToPath(id, isLabeled)}");
                }
                if (box == null)
                {
                    continue; //for training, we need to know the box shape for each picture to remove it while training
                }
                textLabel = idToTextLabel[id];
            }

            elementIdToPaths.Add(new List<string> { EffiSciences95Utils.IdToPath(id, isLabeled) });
            elementIdToDescription.Add(id.ToString());
            elementIdToCategoryIndex.Add(textLabel);
            boxes.Add(box);
            if (maxElementCount != -1 && elementIdToPaths.Count >= maxElementCount)
            {
                break;
            }
        }

        return new EffiSciences95DirectoryDataSet(
            isLabeled,
            boxes,
            elementIdToPaths,
            elementIdToDescription,
            elementIdToCategoryIndex,
            null
        );
    }

    private static readonly List<Tuple<float, float>> PrecomputedMeanAndVolatilityForEachChannel = new()
    {
        Tuple.Create(128.42516f, 79.42157f),
        Tuple.Create(107.48822f, 74.195564f),
        Tuple.Create(97.46115f, 73.76817f)
    };



    private EffiSciences95DirectoryDataSet(
        bool isLabeled,
        List<EffiSciences95Row> boxes,
        List<List<string>> elementIdToPaths,
        List<string> elementIdToDescription,
        List<int> elementIdToCategoryIndex,
        CpuTensor<float> expectedYIfAny
    )
        : base(elementIdToPaths, 
            elementIdToDescription, 
            elementIdToCategoryIndex, 
            expectedYIfAny,
            EffiSciences95Utils.NAME, 
            Objective_enum.Classification, 
            3,
            new []{"old", "young"},
            PrecomputedMeanAndVolatilityForEachChannel,
            ResizeStrategyEnum.None, 
            null)
    {
        _boxes = boxes;
    }


    public override BitmapContent OriginalElementContent(int elementId, int targetHeight, int targetWidth, bool withDataAugmentation, bool isTraining)
    {
        var res = base.OriginalElementContent(elementId, targetHeight, targetWidth, withDataAugmentation, isTraining);
        if (res == null || !isTraining || _boxes[elementId] == null || _boxes[elementId].IsEmpty)
        {
            return res;
        }

        //we need to draw a black box on the picture
        var boxToRemove0 = _boxes[elementId];

        int row_Start = boxToRemove0.Row_start;
        int row_End = boxToRemove0.Row_start + boxToRemove0.Height - 1;
        int col_Start = boxToRemove0.Col_start;
        int width = boxToRemove0.Width;

        if (boxToRemove0.Label == "o")
        {
            var otherBox = FindBiggerBoxWithLabels(boxToRemove0, "y", 20);
            if (otherBox != null)
            {
                row_Start -= (otherBox.Height - boxToRemove0.Height) / 2;
                row_End = row_Start + otherBox.Height - 1;
                col_Start -= (otherBox.Width - boxToRemove0.Width)/2;
                width = otherBox.Width;
            }
        }

        row_Start -= _r.Next(3);
        row_End += _r.Next(3);
        col_Start -= _r.Next(3);
        width += _r.Next(6);

        row_Start = Math.Max(row_Start, 0);
        row_End = Math.Min(row_End, 217);
        col_Start = Math.Max(col_Start, 0);
        width = Math.Min(width, 178 - col_Start);




        var span = res.SpanContent;
        for (int c = 0; c < Channels; ++c)
        {
            for (int row = row_Start; row <= row_End; ++row)
            {
                int idx = res.Idx(c, row, col_Start);
                span.Slice(idx, width).Clear();
            }
        }
        //res.Save("c:/temp/toto_"+ ElementIdToDescription(elementId)+".png");
        return res;
    }


    private EffiSciences95Row FindBiggerBoxWithLabels(EffiSciences95Row box, string mandatoryLabel, int remainingTries)
    {
        for (int i = 0; i <= remainingTries; ++i)
        {

            var otherBox = _boxes[_r.Next(_boxes.Count)];
            if (otherBox == null || otherBox.IsEmpty || otherBox.Label != mandatoryLabel)
            {
                continue;
            }

            if (otherBox.Width >= box.Width && otherBox.Height >= box.Height)
            {
                return otherBox;
            }
        }

        return null;

    }
}