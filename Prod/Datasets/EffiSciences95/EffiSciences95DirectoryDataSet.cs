using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using SharpNet.CPU;
using SharpNet.Pictures;

namespace SharpNet.Datasets.EffiSciences95;

public class EffiSciences95DirectoryDataSet : DirectoryDataSet
{
    private readonly List<EffiSciences95Row> _boxes;
    private readonly EffiSciences95DatasetSample _datasetSample;

    private readonly Random _r = new (Utils.RandomSeed());

    public static EffiSciences95DirectoryDataSet ValueOf(EffiSciences95DatasetSample datasetSample, bool isLabeled, int maxElementCount = -1)
    {
        //maxElementCount = 100; //!D

        var idToBoxes = new EffiSciences95BoxesDataset(isLabeled).Content;
        // ReSharper disable once ConditionIsAlwaysTrueOrFalse
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
                    // ReSharper disable once ConditionIsAlwaysTrueOrFalse
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
            datasetSample,
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
        EffiSciences95DatasetSample datasetSample,
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
        _datasetSample = datasetSample;
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
        var rectBoxToRemove = boxToRemove0.Shape;

        if (boxToRemove0.Label == "o" && _datasetSample.EnlargeOldBoxToYoungBoxShape)
        {
            var otherBox = FindBiggerBoxWithLabels(boxToRemove0, "y", 20);
            if (otherBox != null)
            {
                rectBoxToRemove.Y -= (otherBox.Height - boxToRemove0.Height) / 2;
                rectBoxToRemove.Height = otherBox.Height;
                rectBoxToRemove.X -= (otherBox.Width - boxToRemove0.Width) / 2;
                rectBoxToRemove.Width = otherBox.Width;
            }
        }
        ClearBitmap(res, rectBoxToRemove);

        if (_datasetSample.AddNewBoxOfOtherCategory)
        {
            var otherCategoryBox = RandomBoxWithLabels(boxToRemove0.Label == "o"?"y":"o", 20);
            if (otherCategoryBox != null)
            {
                ClearBitmap(res, otherCategoryBox.Shape);
            }
        }
        return res;
    }


    public void ClearBitmap(BitmapContent res, Rectangle rect)
    {
        Debug.Assert(_datasetSample.MaxEnlargeForBox >= _datasetSample.MinEnlargeForBox);
        if (_datasetSample.MaxEnlargeForBox > 0)
        {
            int min = _datasetSample.MinEnlargeForBox;
            int max = _datasetSample.MaxEnlargeForBox;
            int dy = _r.Next(min, max+1);
            rect.Y -= dy;
            rect.Height += dy+_r.Next(min, max + 1);
            int dx = _r.Next(min, max + 1);
            rect.X -= dx;
            rect.Width += dx + _r.Next(min, max + 1);
        }

        var row_Start = Math.Max(rect.Top, 0);
        var row_End = Math.Min(rect.Bottom-1, 217);
        var col_Start = Math.Max(rect.Left, 0);
        var col_End = Math.Min(rect.Right - 1, 177);
        var width = col_End - col_Start + 1;

        var span = res.SpanContent;
        for (int c = 0; c < Channels; ++c)
        {
            //!D TO CHECK: var meanValue = (byte)PrecomputedMeanAndVolatilityForEachChannel[c].Item1;
            for (int row = row_Start; row <= row_End; ++row)
            {
                int idx = res.Idx(c, row, col_Start);
                span.Slice(idx, width).Clear();
                //TODO: TO CHECK: for (int col = idx; col < idx + width; ++col) {span[col] = meanValue;}
            }
        }
    }


    private EffiSciences95Row RandomBoxWithLabels(string mandatoryLabel, int remainingTries)
    {
        for (int i = 0; i <= remainingTries; ++i)
        {
            var otherBox = _boxes[_r.Next(_boxes.Count)];
            if (otherBox != null && !otherBox.IsEmpty && otherBox.Label == mandatoryLabel)
            {
                return otherBox;
            }
        }
        return null;

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