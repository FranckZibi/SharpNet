using System;
using System.Collections.Generic;
using System.Drawing;
using SharpNet.Pictures;

namespace SharpNet.Datasets.EffiSciences95;

/// <summary>
/// this class is used to load each picture of a dataset (see method: OriginalElementContent).
/// Before returning the picture, it will remove the label (old or young) appearing in the picture
/// </summary>
public class EffiSciences95DirectoryDataSet : DirectoryDataSet
{
    private readonly List<EffiSciences95LabelCoordinates> _labelCoordinates;
    private readonly EffiSciences95DatasetSample _datasetSample;
    private readonly Random _r = new (Utils.RandomSeed());


    public static EffiSciences95DirectoryDataSet ValueOf(EffiSciences95DatasetSample datasetSample, string directory, int maxElementCount = -1)
    {
        //we load the coordinates of the box containing the label ("old" or "young") for each picture (as computed in method FindLabelCoordinates in class LabelFinder)
        var idToLabelCoordinatesInPicture = new EffiSciences95LabelCoordinatesDataset(directory).Content;
        // the known label from the Labeled (train) dataset
        var pictureId_to_TrainLabel = (directory == "Labeled")?EffiSciences95Utils.LoadPredictionFile("Labeled_TextTarget.csv") :null;
        
        List<List<string>> elementIdToPaths = new();
        List<string> elementId_to_pictureId = new();
        List<int> elementId_to_label = new();
        List<EffiSciences95LabelCoordinates> allLabelCoordinatesInPictures = new();

        for(int pictureId=0;pictureId<=EffiSciences95Utils.MaxPictureId(directory);++pictureId)
        {
            var labelCoordinatesInPicture = idToLabelCoordinatesInPicture.TryGetValue(pictureId, out var tmp) ? tmp : null;
            if (labelCoordinatesInPicture!= null && (labelCoordinatesInPicture.IsEmpty || !labelCoordinatesInPicture.HasKnownLabel))
            {
                labelCoordinatesInPicture = null;
            }
            // for Labeled (train) dataset:
            //    0 or 1
            // for Unlabeled (test) dataset:
            //    -1
            int label = -1; 
            if (directory=="Labeled")
            {
                if (labelCoordinatesInPicture == null)
                {
                    continue; //for training, we need to know the box shape for each picture to remove it while training
                }
                // we extract the label for this training pictureId
                label = pictureId_to_TrainLabel[pictureId];
            }
            elementIdToPaths.Add(new List<string> { EffiSciences95Utils.PictureIdToPath(pictureId, directory) });
            elementId_to_pictureId.Add(pictureId.ToString());
            elementId_to_label.Add(label);
            allLabelCoordinatesInPictures.Add(labelCoordinatesInPicture);
            if (maxElementCount != -1 && elementIdToPaths.Count >= maxElementCount)
            {
                break;
            }
        }

        return new EffiSciences95DirectoryDataSet(
            datasetSample,
            allLabelCoordinatesInPictures,
            elementIdToPaths,
            elementId_to_pictureId,
            elementId_to_label);
    }

    private EffiSciences95DirectoryDataSet(
        EffiSciences95DatasetSample datasetSample,
        List<EffiSciences95LabelCoordinates> labelCoordinates,
        List<List<string>> elementIdToPaths,
        List<string> y_IDs,
        List<int> elementIdToCategoryIndex
    )
        : base(
            datasetSample,
            elementIdToPaths, 
            elementIdToCategoryIndex, 
            null,
            EffiSciences95Utils.NAME, 
            PrecomputedMeanAndVolatilityForEachChannel,
            ResizeStrategyEnum.None, 
            null,
            y_IDs.ToArray())
    {
        _datasetSample = datasetSample;
        _labelCoordinates = labelCoordinates;
    }



    /// <summary>
    /// return a picture (from the training (Labeled) or test (Unlabeled) dataset) after removing the box that seems to contain the label 
    /// </summary>
    /// <returns></returns>
    public override BitmapContent OriginalElementContent(int elementId, int channels, int targetHeight, int targetWidth, bool withDataAugmentation, bool isTraining)
    {
        var res = base.OriginalElementContent(elementId, channels, targetHeight, targetWidth, withDataAugmentation, isTraining);
        if (res == null || !isTraining || _labelCoordinates[elementId] == null || _labelCoordinates[elementId].IsEmpty)
        {
            return res;
        }

        //we need to draw a black box on the picture
        var labelCoordinateToRemove0 = _labelCoordinates[elementId];
        var rectBoxToRemove = labelCoordinateToRemove0.Shape;

        var otherLabelCoordinate = FindBiggerBoxWithLabels(labelCoordinateToRemove0);
        if (otherLabelCoordinate != null)
        {
            rectBoxToRemove.Y -= (otherLabelCoordinate.Height - labelCoordinateToRemove0.Height) / 2;
            rectBoxToRemove.Height = otherLabelCoordinate.Height;
            rectBoxToRemove.X -= (otherLabelCoordinate.Width - labelCoordinateToRemove0.Width) / 2;
            rectBoxToRemove.Width = otherLabelCoordinate.Width;
        }
        ClearBitmap(res, rectBoxToRemove);
        return res;
    }

    private void ClearBitmap(BitmapContent res, Rectangle rect)
    {
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
        for (int c = 0; c < res.GetChannels(); ++c)
        {
            for (int row = row_Start; row <= row_End; ++row)
            {
                int idx = res.Idx(c, row, col_Start);
                span.Slice(idx, width).Clear();
            }
        }
    }

    /// <summary>
    /// return an element from the training dataset containing "young" and with a box label bigger than the one in 'box'
    /// </summary>
    /// <returns></returns>
    private EffiSciences95LabelCoordinates FindBiggerBoxWithLabels(EffiSciences95LabelCoordinates box)
    {
        for (int i = 0; i <= 20; ++i)
        {
            //we look for a bigger box containing "young" from the training (Labeled) dataset
            var otherLabelCoordinate = _labelCoordinates[_r.Next(_labelCoordinates.Count)];
            if (otherLabelCoordinate == null || otherLabelCoordinate.IsEmpty || otherLabelCoordinate.Label != "y")
            {
                continue;
            }
            if (otherLabelCoordinate.Width >= box.Width && otherLabelCoordinate.Height >= box.Height)
            {
                return otherLabelCoordinate;
            }
        }
        return null;
    }

    private static readonly List<Tuple<float, float>> PrecomputedMeanAndVolatilityForEachChannel = new()
        {
            Tuple.Create(128.42516f, 79.42157f),
            Tuple.Create(107.48822f, 74.195564f),
            Tuple.Create(97.46115f, 73.76817f)
        };
}
