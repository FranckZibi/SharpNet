using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using JetBrains.Annotations;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.Networks;

namespace SharpNet.Datasets.CFM60;


public interface TimeSeriesSinglePoint
{
    string UniqueId { get; }
    string TimeSeriesFamily { get; }
    float TimeSeriesTimeStamp { get; }
    float ExpectedTarget { get; } //y
}

public interface IGetDatasetSample
{
    AbstractDatasetSample GetDatasetSample();
}

public class TimeSeriesDataset : DataSet, ITimeSeriesDataSet, IGetDatasetSample
{
    public EncoderDecoder_NetworkSample EncoderDecoder_NetworkSample { get; }

    // CFM60EntryID = CFM60Entry.ID: the unique ID of a CFM60Entry
    // elementId : id of an element in the dataSet (in range [0, dataSet.Count[ )
    private readonly Dictionary<string, float> _idToPrediction = new ();

    public override CpuTensor<float> Y { get; }

    public DatasetSampleForTimeSeries DatasetSampleForTimeSeries { get; }

    private readonly TimeSeriesDataset _trainingDataSetOldIfAny;

    public override string to_csv_in_directory(string directory, bool addTargetColumnAsFirstColumn, bool includeIdColumns, bool overwriteIfExists)
    {
        //!D we do nothing
        return "";
    }



    public override DataFrame ExtractIdDataFrame()
    {
        var ids = Enumerable.Range(0, _elementIdToLastAssociateEntry.Count).Select(i => _elementIdToLastAssociateEntry[i].UniqueId).ToArray();
        return DataFrame.New(ids, IdColumns);
    }



    //TODO : add element for each valid day, even if no entry is associated with that day
    private readonly Dictionary<string, List<TimeSeriesSinglePoint>> _pidToSortedEntries = new();
    private readonly Dictionary<string, int> _EntryUniqueIdToIndexIn_pidToSortedEntries = new();
    private readonly Dictionary<int, TimeSeriesSinglePoint> _elementIdToLastAssociateEntry = new();
    [NotNull] public readonly TimeSeriesSinglePoint[] Entries;

    public bool IsTrainingDataSet => _trainingDataSetOldIfAny == null;
    int EntriesCountForEachElementId_X => EncoderDecoder_NetworkSample.Use_Decoder ? EncoderDecoder_NetworkSample.Encoder_TimeSteps : 1 + EncoderDecoder_NetworkSample.Encoder_TimeSteps;
    int EntriesCountForEachElementId_Y => EncoderDecoder_NetworkSample.Use_Decoder ? EncoderDecoder_NetworkSample.Decoder_TimeSteps : 1;
    public int Total_TimeSteps => EncoderDecoder_NetworkSample.Total_TimeSteps;

    public override int Count => Y.Shape[0];
    public override int ElementIdToCategoryIndex(int elementId)
    {
        return -1;
    }
    public IEnumerable<TimeSeriesSinglePoint> ElementId_to_YEntries(int elementId)
    {
        var lastEntry = _elementIdToLastAssociateEntry[elementId];
        var pidEntries = _pidToSortedEntries[lastEntry.TimeSeriesFamily];
        var lastIdx = _EntryUniqueIdToIndexIn_pidToSortedEntries[lastEntry.UniqueId];
        int firstIdx = lastIdx - EntriesCountForEachElementId_Y + 1;
        for (int idx = firstIdx; idx <= lastIdx; ++idx)
        {
            yield return pidEntries[idx];
        }
    }
    
    public override List<int[]> XMiniBatch_Shape(int[] shapeForFirstLayer)
    {
        var result = new List<int[]> { shapeForFirstLayer };
        if (EncoderDecoder_NetworkSample.Use_Decoder)
        {
            var inputShapeDecoder = new[]
            {
                shapeForFirstLayer[0],
                EncoderDecoder_NetworkSample.Decoder_TimeSteps,
                DatasetSampleForTimeSeries.GetInputSize(false)
            };
            result.Add(inputShapeDecoder);
        }
        return result;
    }

    public override ITrainingAndTestDataset IntSplitIntoTrainingAndValidation(int countInTrainingSet)
    {
        return SplitIntoTrainingAndValidation(((float)countInTrainingSet) / Entries.Length);
    }
    public override ITrainingAndTestDataset SplitIntoTrainingAndValidation(double percentageInTrainingSet)
    {
        float dayThreshold = DayThreshold(Entries, percentageInTrainingSet);
        var training = new TimeSeriesDataset(
            Name,
            Entries.Where(e => e.TimeSeriesTimeStamp <= dayThreshold).ToArray(),
            EncoderDecoder_NetworkSample,
            DatasetSampleForTimeSeries,
            null);
        var validation = new TimeSeriesDataset(
            Name,
            Entries.Where(e => e.TimeSeriesTimeStamp > dayThreshold).ToArray(),
            EncoderDecoder_NetworkSample,
            DatasetSampleForTimeSeries,
            training);
        return new TrainingAndTestDataset(training, validation, Name);
    }
    public override DataSet SubDataSet(double percentageToKeep)
    {
        throw new NotImplementedException();
    }

    public static float DayThreshold(IList<TimeSeriesSinglePoint> entries, double percentageInTrainingSet)
    {
        var sortedDays = entries.Select(e => e.TimeSeriesTimeStamp).OrderBy(x => x).ToArray();
        var countInTraining = (int)(percentageInTrainingSet * entries.Count);
        var dayThreshold = sortedDays[countInTraining];
        return dayThreshold;
    }

    protected override int GetMaxElementsToLoad(int[] shuffledElementId, int firstIndexInShuffledElementId, int batchSize)
    {
        var defaultResult = base.GetMaxElementsToLoad(shuffledElementId, firstIndexInShuffledElementId, batchSize);
        if (IsTrainingDataSet)
        {
            return defaultResult;
        }

        //in Validation & Test DataSet, we can only make at most 1 prediction / pid in each mini batch
        var observedPid = new HashSet<string>();
        for (int indexInShuffledElementId = firstIndexInShuffledElementId; indexInShuffledElementId < shuffledElementId.Length; ++indexInShuffledElementId)
        {
            int currentLength = indexInShuffledElementId - firstIndexInShuffledElementId;
            if (currentLength >= batchSize)
            {
                return batchSize;
            }
            var elementId = shuffledElementId[indexInShuffledElementId];
            var pid = _elementIdToLastAssociateEntry[elementId].TimeSeriesFamily;
            if (!observedPid.Add(pid))
            {
                //we have already see this pid before.
                //We can only make one prediction / pid in each batch
                return currentLength;
            }
        }
        return defaultResult;
    }


    public void SetBatchPredictionsForInference(int[] batchElementIds, Tensor batchPredictions)
    {
        Debug.Assert(batchPredictions.Count == batchElementIds.Length * EntriesCountForEachElementId_Y);
        var predictions = batchPredictions.ContentAsFloatArray();
        int nextPredictionIdx = 0;
        foreach (var elementId in batchElementIds)
        {
            foreach (var e in ElementId_to_YEntries(elementId))
            {
                _idToPrediction[e.UniqueId] = predictions[nextPredictionIdx++];
            }
        }
        Debug.Assert(nextPredictionIdx == batchPredictions.Count);
    }

    public TimeSeriesDataset(string name, TimeSeriesSinglePoint[] entries, EncoderDecoder_NetworkSample networkSample, 
        DatasetSampleForTimeSeries datasetSample, TimeSeriesDataset trainingDataSetOldIfAny = null)
        : base(name,
            datasetSample.GetObjective(),
            networkSample.Encoder_TimeSteps,
            null,
            ResizeStrategyEnum.None,
            datasetSample.GetColumnNames(),
            datasetSample.CategoricalFeatures,
            datasetSample.IdColumns,
            UseBackgroundThreadToLoadNextMiniBatchV2(trainingDataSetOldIfAny),
            ',')
    {
        EncoderDecoder_NetworkSample = networkSample;
        DatasetSampleForTimeSeries = datasetSample;
        _trainingDataSetOldIfAny = trainingDataSetOldIfAny;
        Entries = entries;
        int elementId = 0;

        //we initialize: _pidToSortedEntries
        foreach (var entry in Entries.OrderBy(e => e.TimeSeriesFamily).ThenBy(e => e.TimeSeriesTimeStamp))
        {
            if (!_pidToSortedEntries.ContainsKey(entry.TimeSeriesFamily))
            {
                _pidToSortedEntries[entry.TimeSeriesFamily] = new List<TimeSeriesSinglePoint>();
            }

            _pidToSortedEntries[entry.TimeSeriesFamily].Add(entry);
        }

        //we initialize _IDToIndexIn_pidToSortedEntries
        foreach (var e in _pidToSortedEntries.Values)
        {
            for (int index_in_pidToSortedEntries = 0; index_in_pidToSortedEntries < e.Count; ++index_in_pidToSortedEntries)
            {
                _EntryUniqueIdToIndexIn_pidToSortedEntries[e[index_in_pidToSortedEntries].UniqueId] = index_in_pidToSortedEntries;
            }
        }

        //we initialize _elementIdToAssociateLastCFM60Entry
        int longestEntry = _pidToSortedEntries.Values.Select(x => x.Count).Max();
        var pids = _pidToSortedEntries.Keys.OrderBy(x => x).ToArray();
        int idxLastEntry = IsTrainingDataSet
            ? EntriesCountForEachElementId_X + EntriesCountForEachElementId_Y - 1
            : EntriesCountForEachElementId_Y - 1;
        while (idxLastEntry < longestEntry)
        {
            foreach (var pid in pids)
            {
                var pidEntries = _pidToSortedEntries[pid];
                if (idxLastEntry < pidEntries.Count)
                {
                    _elementIdToLastAssociateEntry[elementId] = pidEntries[idxLastEntry];
                    ++elementId;
                }
            }

            if (IsTrainingDataSet)
            {
                //in the Training DataSet: only entries in the range [TimeSteps, +infinite[ can be trained
                idxLastEntry += 1;
            }
            else
            {
                //in the Validation/Test DataSets: each element is a prediction to make
                idxLastEntry += EntriesCountForEachElementId_Y;
            }
        }

        //we initialize Y
        //total number of items in the dataSet
        int count = elementId;
        var yData = new float[count * EntriesCountForEachElementId_Y];
        int nextIdxInY = 0;
        for (elementId = 0; elementId < count; ++elementId)
        {
            foreach (var e in ElementId_to_YEntries(elementId))
            {
                yData[nextIdxInY++] = e.ExpectedTarget;
            }
        }
        Debug.Assert(nextIdxInY == yData.Length);
        Y = new CpuTensor<float>(new[] { count, EntriesCountForEachElementId_Y }, yData);

        //if we are in a training data set
        if (trainingDataSetOldIfAny == null)
        {
            //we ensure that the training data set is valid
            foreach (var (pid, trainingEntries) in _pidToSortedEntries)
            {
                if (trainingEntries.Count <= Total_TimeSteps)
                {
                    throw new Exception("invalid Training DataSet: not enough entries (" + trainingEntries.Count + ") for pid " + pid);
                }

                if (trainingEntries.Any(x => float.IsNaN(x.ExpectedTarget)))
                {
                    throw new Exception("invalid Training DataSet: no known Y value for pid " + pid);
                }
            }
        }
        else //validation or test data set
        {
            //we ensure that the associate training data set is valid
            foreach (var (pid, validationEntries) in _pidToSortedEntries)
            {
                if (!trainingDataSetOldIfAny._pidToSortedEntries.ContainsKey(pid))
                {
                    throw new Exception("validation pid " + pid + " doesn't exist in training data set");
                }
                if (validationEntries.Count == 0)
                {
                    throw new Exception("not enough entries (" + validationEntries.Count + ") in validation data set for pid " + pid);
                }
            }
        }

    }
    private static bool UseBackgroundThreadToLoadNextMiniBatchV2(TimeSeriesDataset trainingDataSetOldIfAny)
    {
        if (trainingDataSetOldIfAny != null)
        {
            //for Validation/Test DataSet, we should not use a background thread for loading next mini batch data
            return false;
        }
        //for Training DataSet, we should use background thread for loading next mini batch
        return true;
    }


    /// <summary>
    /// return the entry associated with the pid 'pid' for index 'indexInPidEntryArray'
    /// </summary>
    /// <param name="timeSeriesFamily">the pid to retrieve</param>
    /// <param name="indexInPidEntryArray">
    /// index of the entry to retrieve.
    /// if strictly less then 0:
    ///     it means that we are in a validation DataSet and we want to load data from
    ///     the associated Training DataSet
    /// </param>
    /// <returns></returns>
    private TimeSeriesSinglePoint GetEntry(string timeSeriesFamily, int indexInPidEntryArray)
    {
        if (indexInPidEntryArray >= 0)
        {
            return _pidToSortedEntries[timeSeriesFamily][indexInPidEntryArray];
        }
        if (IsTrainingDataSet)
        {
            throw new Exception("no Entry for pid " + timeSeriesFamily + " at index=" + indexInPidEntryArray);
        }
        var trainingEntries = _trainingDataSetOldIfAny._pidToSortedEntries[timeSeriesFamily];
        return trainingEntries[trainingEntries.Count + indexInPidEntryArray];
    }


    /// <summary>
    /// 
    /// </summary>
    /// <param name="elementId">gives the entry to predict</param>
    /// <param name="indexInBuffer"></param>
    /// <param name="all_xBuffer">
    ///     input shape: (batchSize, TimeSteps, InputSize)</param>
    /// <param name="yBuffer">
    ///     output shape: (batchSize, 1)
    /// </param>
    /// <param name="withDataAugmentation"></param>
    /// <param name="isTraining"></param>
    protected override void LoadAt(int elementId, int indexInBuffer, List<CpuTensor<float>> all_xBuffer,
        CpuTensor<float> yBuffer, bool withDataAugmentation, bool isTraining)
    {
        var xEncoder = all_xBuffer[0];
        Debug.Assert(xEncoder.Shape.Length == 3);
        Debug.Assert(indexInBuffer >= 0 && indexInBuffer < all_xBuffer[0].Shape[0]);
        Debug.Assert(xEncoder.Shape[1] == EncoderDecoder_NetworkSample.Encoder_TimeSteps);
        Debug.Assert(xEncoder.Shape[2] == DatasetSampleForTimeSeries.GetInputSize(true));
        Debug.Assert(yBuffer == null || all_xBuffer[0].Shape[0] == yBuffer.Shape[0]); //same batch size
        Debug.Assert(yBuffer == null || yBuffer.SameShapeExceptFirstDimension(Y.Shape));
        Debug.Assert(yBuffer == null || yBuffer.SameShapeExceptFirstDimension(Y.Shape));

        LoadAt(elementId, indexInBuffer, xEncoder, true);
        if (EncoderDecoder_NetworkSample.Use_Decoder)
        {
            Debug.Assert(all_xBuffer.Count == 2);
            var xDecoder = all_xBuffer[1];
            LoadAt(elementId, indexInBuffer, xDecoder, false);
        }

        if (yBuffer != null)
        {
            Y.CopyTo(Y.Idx(elementId), yBuffer, yBuffer.Idx(indexInBuffer), yBuffer.MultDim0);
        }
    }
    public override void LoadAt(int elementId, int indexInBuffer, CpuTensor<float> xBuffer, CpuTensor<float> yBuffer,
        bool withDataAugmentation, bool isTraining)
    {
        throw new Exception("should never be called");
    }
    

    private void LoadAt(int elementId, int indexInBuffer, CpuTensor<float> x, bool isEncoder)
    {
        var xElementId = x.AsFloatCpuSpan.Slice(indexInBuffer * x.MultDim0, x.MultDim0);

        int idx = 0;
        var lastAssociateCFM60Entry = _elementIdToLastAssociateEntry[elementId];
        var pid = lastAssociateCFM60Entry.TimeSeriesFamily;
        int lastIndexInPidEntries = _EntryUniqueIdToIndexIn_pidToSortedEntries[lastAssociateCFM60Entry.UniqueId];
        Debug.Assert(_pidToSortedEntries[pid][lastIndexInPidEntries].UniqueId == lastAssociateCFM60Entry.UniqueId);
        int expectedInputSize = DatasetSampleForTimeSeries.GetInputSize(isEncoder);

        if (isEncoder && EncoderDecoder_NetworkSample.Use_Decoder)
        {
            Debug.Assert(EncoderDecoder_NetworkSample.Decoder_TimeSteps >= 1);
            lastIndexInPidEntries -= EncoderDecoder_NetworkSample.Decoder_TimeSteps;
        }

        int timeSteps = x.Shape[1];
        // ReSharper disable once LoopVariableIsNeverChangedInsideLoop
        for (int timeStep = 0; timeStep < timeSteps; ++timeStep)
        {
            var indexInPidEntryArray = lastIndexInPidEntries - timeSteps + timeStep + 1;
            var entry = GetEntry(pid, indexInPidEntryArray);


            float prev_Y = float.NaN;
            //y estimate
            if (DatasetSampleForTimeSeries.Use_prev_Y && isEncoder)
            {
                var indexOfyEntryInPidEntryArray = EncoderDecoder_NetworkSample.Use_Decoder
                    ? indexInPidEntryArray
                    : indexInPidEntryArray - 1;  //we take the previous entry
                var yEntry = GetEntry(pid, indexOfyEntryInPidEntryArray);
                if (IsTrainingDataSet
                    || indexOfyEntryInPidEntryArray < 0 //the entry is in the training set
                   )
                {
                    //we will use the true value for Y
                    var y = yEntry.ExpectedTarget;
                    if (float.IsNaN(y))
                    {
                        throw new Exception("no Y value associated with entry " + (indexInPidEntryArray - 1) + " of pid " + pid);
                    }
                    prev_Y = y;
                }
                else
                {
                    //we need to use the estimated value for Y (even if the true value of Y is available)
                    if (!_idToPrediction.ContainsKey(yEntry.UniqueId))
                    {
                        throw new Exception("missing prediction for UniqueId " + yEntry.UniqueId + " with pid " + pid + " : it is required to make the prediction for next UniqueId " + entry.UniqueId);
                    }
                    prev_Y = _idToPrediction[yEntry.UniqueId];
                }
            }

            int count =DatasetSampleForTimeSeries.LoadEntry(entry, prev_Y, xElementId, idx, EncoderDecoder_NetworkSample, isEncoder);
            if (count != expectedInputSize)
            {
                throw new Exception("expecting " + expectedInputSize + " elements but got " + count);
            }
            idx += count;
        }
    }

    public AbstractDatasetSample GetDatasetSample()
    {
        return DatasetSampleForTimeSeries;
    }
}