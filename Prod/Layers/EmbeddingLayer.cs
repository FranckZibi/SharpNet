using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using JetBrains.Annotations;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.Networks;
using SharpNet.Optimizers;

namespace SharpNet.Layers;

/// <summary>
/// This layer can only be used as the second layer in a model (the first layer being the InputLayer).
/// 
/// =======================================================================================================
/// input 'x' shape                                     output 'y' shape
/// =======================================================================================================
/// (batchSize, timeSteps)                              (batchSize, timeSteps, EmbeddingDim)
/// =======================================================================================================
/// (batchSize, input_length)                           (batchSize, input_length+EmbeddingDim-1)
/// =======================================================================================================
/// (batchSize, timeSteps, input_length)                (batchSize, timeSteps, input_length+EmbeddingDim-1)
/// =======================================================================================================
/// </summary>
public sealed class EmbeddingLayer : Layer
{
    #region Private fields
        
    #region trainable parameters
    /// <summary>
    /// Word Embedding, of shape: (VocabularySize, EmbeddingDim)
    /// </summary>
    [NotNull] private Tensor _weights;
    #endregion
        
    #region gradients
    /// <summary>
    /// same shape as '_weights'
    /// </summary>
    [NotNull] private Tensor _weightGradients;
    /// <summary>
    /// Adam or SGD optimizer or Vanilla SGD
    /// </summary>
    #endregion

    [NotNull] private readonly Optimizer _optimizer;


    /// <summary>
    /// each element is the description of an embedding:
    ///     vocabularySize:
    ///         Size of the vocabulary, i.e. maximum integer index + 1
    ///         In the input 'x' tensor:
    ///         each wordIndex element must be in [0, VocabularySize-1]
    ///     embeddingDim:
    ///         Dimension of the dense embedding
    ///     indexInLastDimensionToUse:
    ///         index in last dimension of input tensor where to find the index of the embedding to use
    /// </summary>
    private readonly List<(int vocabularySize, int embeddingDim, int indexInLastDimensionToUse)> _embeddingDescriptions;


    
    /// <summary>
    /// regularization hyper parameter. 0 if no L2 regularization
    /// </summary>
    private readonly double LambdaL2Regularization;
    /// <summary>
    /// if value > 0 
    ///     clip values of weights gradients in range [-ClipValueForGradients, ClipValueForGradients]
    /// else
    ///     do not clip values
    /// </summary>
    private readonly float ClipValueForGradients;
    /// <summary>
    /// true if we should divide the weight gradients by the time steps
    /// </summary>
    private readonly bool DivideGradientsByTimeSteps;
    #endregion



    public static List<(int vocabularySize, int embeddingDim, int indexInLastDimensionToUse)> ToEmbeddingLayerDescription(
        int[] vocabularySizes,
        int[] embeddingDims,
        int[] indexesInLastDimensionToUse)
    {
        List<(int vocabularySize, int embeddingDim, int indexInLastDimensionToUse)> result = new();
        if (vocabularySizes.Length != embeddingDims.Length || vocabularySizes.Length != indexesInLastDimensionToUse.Length)
        {
            throw new ArgumentException($"input are not the same length : {vocabularySizes.Length} vs {embeddingDims.Length} vs {indexesInLastDimensionToUse.Length}");
        }
        for (int i = 0; i < vocabularySizes.Length; i++)
        {
            result.Add((vocabularySizes[i], embeddingDims[i], indexesInLastDimensionToUse[i]));
        }
        return result;
    }

    #region constructor
    public EmbeddingLayer(
        IEnumerable<(int, int, int)> embeddingDescriptions,
        double lambdaL2Regularization,
        float clipValueForGradients,
        bool divideGradientsByTimeSteps,
        bool trainable, Network network, string layerName) : base(network, layerName)
    {
        _embeddingDescriptions = embeddingDescriptions.OrderBy(t => t.Item3).ToList();
        if (_embeddingDescriptions[0].indexInLastDimensionToUse < 0 && _embeddingDescriptions.Count != 1)
        {
            throw new ArgumentException($"only 1 element is allowed if indexesInLastDimensionToUse = {_embeddingDescriptions[0].indexInLastDimensionToUse}");
        }
        LambdaL2Regularization = lambdaL2Regularization;
        ClipValueForGradients = clipValueForGradients;
        DivideGradientsByTimeSteps = divideGradientsByTimeSteps;

        Trainable = trainable;

        //trainable params
        int weightColumns = _embeddingDescriptions.Select(t=>t.vocabularySize*t.embeddingDim).Sum();
        _weights = GetFloatTensor(new[] { 1, weightColumns });
        _weightGradients = GetFloatTensor(_weights.Shape);

        _optimizer = GetOptimizer(_weights.Shape, null);
        ResetParameters(false);
    }
    #endregion

    #region forward and backward propagation
    public override void ForwardPropagation(List<Tensor> allX, Tensor y, bool isTraining)
    {
        ++NbForwardPropagation;
        Debug.Assert(allX.Count == 1);
        var x = allX[0];
        Debug.Assert(x.Shape[0] == y.Shape[0]); //same batchSize
        Debug.Assert(y.Shape.Length != 3 || x.Shape[1] == y.Shape[1]); //same timeSteps
        Debug.Assert(!ShouldEmbedEachElementOfLastDimension || x.Shape[1] == y.Shape[1]); //same timeSteps
        int deltaForIndexesInLastDimensionToUse = 0;
        var allWeights = Split(_weights);

        var xOriginalShape = (int[])x.Shape.Clone();
        var yOriginalShape = (int[])y.Shape.Clone();

        // we'll ensure that in all cases:
        //  the x shape is (batchSize, timeSteps, input_length)
        //  the y shape is (batchSize, timeSteps, input_length+EmbeddingDim-1)
        if (x.Shape.Length == 2)
        {
            if (ShouldEmbedEachElementOfLastDimension)
            {
                //x shape from (batchSize, timeSteps) to (batchSize, timeSteps, 1)
                x.ReshapeInPlace(new [] { x.Shape[0], x.Shape[1], 1});
            }
            else
            {
                //x shape from (batchSize, input_length) to (batchSize, 1, input_length)
                x.ReshapeInPlace(new [] { x.Shape[0], 1, x.Shape[1] });
                //y shape from (batchSize, input_length+EmbeddingDim-1) to (batchSize, 1, input_length+EmbeddingDim-1)
                y.ReshapeInPlace(new [] { y.Shape[0], 1, y.Shape[1] });
            }
        }

        if (ShouldEmbedEachElementOfLastDimension)
        {
            Debug.Assert(allWeights.Count == 1);
            y.WordEmbeddingForwardPropagation(x, allWeights[0], 0, 0, 0, 0);
        }
        else
        {
            for (var i = 0; i < allWeights.Count; i++)
            {
                var xIndexInLastDimensionToUse = _embeddingDescriptions[i].indexInLastDimensionToUse;
                int copyCountBeforeIndex = (i == 0) ? xIndexInLastDimensionToUse : (xIndexInLastDimensionToUse - _embeddingDescriptions[i-1].indexInLastDimensionToUse - 1);
                int copyCountAfterIndex = (i == allWeights.Count - 1) ? x.Shape[2] - xIndexInLastDimensionToUse - 1 : 0;
                y.WordEmbeddingForwardPropagation(x, allWeights[i], xIndexInLastDimensionToUse, deltaForIndexesInLastDimensionToUse + xIndexInLastDimensionToUse, copyCountBeforeIndex, copyCountAfterIndex);
                deltaForIndexesInLastDimensionToUse += allWeights[i].Shape[1] - 1;
            }
        }

        x.ReshapeInPlace(xOriginalShape);
        y.ReshapeInPlace(yOriginalShape);
    }

    public List<Tensor> Split(Tensor w)
    {
        var res = new List<Tensor>();
        int nextIdxInWeights = 0;
        foreach(var (vocabularySize, embeddingDim, _) in _embeddingDescriptions)
        {
            var shape = new[] { vocabularySize, embeddingDim};
            res.Add(w.Slice(nextIdxInWeights, shape));
            nextIdxInWeights += shape[0] * shape[1];
        }
        return res;
    }


    public int NbForwardPropagation = 0;

    private bool ShouldEmbedEachElementOfLastDimension => _embeddingDescriptions[0].indexInLastDimensionToUse == -1;

    public override void BackwardPropagation(List<Tensor> allX, Tensor y_NotUsed, Tensor dy, List<Tensor> allDx)
    {
        Debug.Assert(y_NotUsed == null);
        Debug.Assert(allX.Count == 1);
        var x = allX[0];
        Debug.Assert(allDx.Count == 1);
        var dx = allDx[0]??GetFloatTensor(x.Shape);

        //we compute dW
        int deltaForIndexesInLastDimensionToUse = 0;
        var allWeightGradients = Split(_weightGradients);


        var xOriginalShape = (int[])x.Shape.Clone();
        var dxOriginalShape = (int[])dx.Shape.Clone();
        var dyOriginalShape = (int[])dy.Shape.Clone();

        // we'll ensure that in all cases:
        //  the x shape is (batchSize, timeSteps, input_length)
        //  the y shape is (batchSize, timeSteps, input_length+EmbeddingDim-1)
        if (x.Shape.Length == 2)
        {
            if (ShouldEmbedEachElementOfLastDimension)
            {
                //x shape from (batchSize, timeSteps) to (batchSize, timeSteps, 1)
                x.ReshapeInPlace(new[] { x.Shape[0], x.Shape[1], 1 });
                dx.ReshapeInPlace(x.Shape);
            }
            else
            {
                //x shape from (batchSize, input_length) to (batchSize, 1, input_length)
                x.ReshapeInPlace(new[] { x.Shape[0], 1, x.Shape[1] });
                dx.ReshapeInPlace(x.Shape);
                //dy shape from (batchSize, input_length+EmbeddingDim-1) to (batchSize, 1, input_length+EmbeddingDim-1)
                dy.ReshapeInPlace(new[] { dy.Shape[0], 1, dy.Shape[1] });
            }
        }

        if (ShouldEmbedEachElementOfLastDimension)
        {
            Debug.Assert(allWeightGradients.Count == 1);
            allWeightGradients[0].WordEmbeddingBackwardPropagation(x, dx, dy, 0, 0, 0, 0);
        }
        else
        {
            for (var i = 0; i < allWeightGradients.Count; i++)
            {
                var dxIndexInLastDimensionToUse = _embeddingDescriptions[i].indexInLastDimensionToUse;
                int copyCountBeforeIndex = (i == 0) ? dxIndexInLastDimensionToUse : (dxIndexInLastDimensionToUse - _embeddingDescriptions[i - 1].indexInLastDimensionToUse - 1);
                int copyCountAfterIndex = (i == allWeightGradients.Count - 1) ? dx.Shape[2] - dxIndexInLastDimensionToUse - 1 : 0;
                allWeightGradients[i].WordEmbeddingBackwardPropagation(x, dx, dy, dxIndexInLastDimensionToUse, deltaForIndexesInLastDimensionToUse + dxIndexInLastDimensionToUse, copyCountBeforeIndex, copyCountAfterIndex);
                deltaForIndexesInLastDimensionToUse += allWeightGradients[i].Shape[1] - 1;
            }
        }

        x.ReshapeInPlace(xOriginalShape);
        dx.ReshapeInPlace(dxOriginalShape);
        dy.ReshapeInPlace(dyOriginalShape);

        if (DivideGradientsByTimeSteps)
        {
            int timeSteps = x.Shape[1];
            _weightGradients.Update_Multiplying_By_Alpha(1f/ timeSteps);
        }

        if (ClipValueForGradients > 1e-6)
        {
            _weightGradients.Clip(-ClipValueForGradients, ClipValueForGradients);
        }

        //L2 regularization on dW
        if (UseL2Regularization)
        {
            int batchSize = dy.Shape[0];
            var alpha = 2 * batchSize * (float)LambdaL2Regularization;
            _weightGradients.Update_Adding_Alpha_X(alpha, _weights);
        }

        if (allDx[0] == null)
        {
            FreeFloatTensor(dx);
        }

    }
    public override bool OutputNeededForBackwardPropagation => false;
    public override bool InputNeededForBackwardPropagation => true;
    #endregion

    #region parameters and gradients
    public override Tensor Weights => _weights;
    public override Tensor WeightGradients => _weightGradients;
    protected override Optimizer Optimizer => _optimizer;
    public override List<Tuple<Tensor, string>> Parameters
    {
        get
        {
            var result = new List<Tuple<Tensor, string>>
            {
                Tuple.Create(_weights, WeightDatasetPath),
            };
            result.RemoveAll(t => t.Item1 == null);
            return result;
        }
    }
    public override void ResetParameters(bool resetAlsoOptimizerWeights = true)
    {
        //trainable params
        _weights.UniformDistribution(Rand, -0.05, +0.05);

        if (resetAlsoOptimizerWeights)
        {
            _optimizer.ZeroMemory();
        }
    }
    public override void ReplaceParameters(List<Tensor> newParameters)
    {
        FreeFloatTensor(ref _weights);
        _weights = newParameters[0];
        Debug.Assert(newParameters.Count == 1);
    }
    public override IDictionary<string, CpuTensor<float>> GetParametersAsCpuFloatTensors(NetworkSample.CompatibilityModeEnum originFramework)
    {
        var result = new Dictionary<string, CpuTensor<float>>();
        result.Add(WeightDatasetPath, _weights.ToCpuFloat());
        return result;
    }
    public override void ReplaceGradients(List<Tensor> newGradients)
    {
        FreeFloatTensor(ref _weightGradients);
        _weightGradients = newGradients[0];
        Debug.Assert(newGradients.Count == 1);
    }

    private string WeightDatasetPath => DatasetNameToDatasetPath("kernel:0");
    #endregion

    #region serialization
    public override string Serialize()
    {
        return RootSerializer()
            .Add(nameof(VocabularySizes), VocabularySizes)
            .Add(nameof(EmbeddingDims), EmbeddingDims)
            .Add(nameof(IndexesInLastDimensionToUse), IndexesInLastDimensionToUse)
            .Add(nameof(LambdaL2Regularization), LambdaL2Regularization)
            .Add(nameof(ClipValueForGradients), ClipValueForGradients)
            .Add(nameof(DivideGradientsByTimeSteps), DivideGradientsByTimeSteps)
            .ToString();
    }

    public int[] VocabularySizes => _embeddingDescriptions.Select(t => t.vocabularySize).ToArray();
    public int[] EmbeddingDims => _embeddingDescriptions.Select(t => t.embeddingDim).ToArray();
    public int[] IndexesInLastDimensionToUse => _embeddingDescriptions.Select(t => t.indexInLastDimensionToUse).ToArray();

    public static EmbeddingLayer Deserialize(IDictionary<string, object> serialized, Network network)
    {
        int[] VocabularySizes = serialized.ContainsKey("VocabularySize")
            ? new[] { (int)serialized["VocabularySize"] }
            : (int[])serialized[nameof(VocabularySizes)];
        int[] EmbeddingDims = serialized.ContainsKey("EmbeddingDim")
            ? new[] { (int)serialized["EmbeddingDim"] }
            : (int[])serialized[nameof(EmbeddingDims)];
        int[] IndexesInLastDimensionToUse = serialized.ContainsKey("IndexInLastDimensionToUse")
            ? new[] { (int)serialized["IndexInLastDimensionToUse"] }
            : (int[])serialized[nameof(IndexesInLastDimensionToUse)];

        return new EmbeddingLayer(
            ToEmbeddingLayerDescription(VocabularySizes, EmbeddingDims, IndexesInLastDimensionToUse),
            (double)serialized[nameof(LambdaL2Regularization)],
            (float)serialized[nameof(ClipValueForGradients)],
            (bool)serialized[nameof(DivideGradientsByTimeSteps)],
            (bool)serialized[nameof(Trainable)],
            network,
            (string)serialized[nameof(LayerName)]);
    }
    public override void AddToOtherNetwork(Network otherNetwork) { AddToOtherNetwork(otherNetwork, Deserialize); }
    #endregion

    public override int[] OutputShape(int batchSize)
    {
        var prevLayerOutputShape = PrevLayer.OutputShape(batchSize);
        var outputShape = (int[])prevLayerOutputShape.Clone();
        outputShape[0] = batchSize;
        if (ShouldEmbedEachElementOfLastDimension)
        {
            Debug.Assert(IndexesInLastDimensionToUse.Length == 1);
            //Debug.Assert(prevLayerOutputShape.Length == 2);
            outputShape = outputShape.Append(EmbeddingDims[0]).ToArray();
            return outputShape;
        }
        else
        {
            //Debug.Assert(prevLayerOutputShape.Length == 3);
            outputShape[^1] += EmbeddingDims.Sum() - EmbeddingDims.Length;
            return outputShape;
        }
    }
    public override string ToString()
    {
        var result = LayerName + ": " + ShapeChangeDescription();
        if (UseL2Regularization)
        {
            result += " with L2Regularization[lambdaValue=" + LambdaL2Regularization + "]";
        }
        result += " " + _weights + " (" + TotalParams + " neurons)";
        return result;
    }

    private bool UseL2Regularization => LambdaL2Regularization > 0.0;
}