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
/// (batchSize, timeSteps)                              (batchSize, timeSteps, embedding_dim)
/// =======================================================================================================
/// (batchSize, input_length)                           (batchSize, input_length+embedding_dim-1)
/// =======================================================================================================
/// (batchSize, timeSteps, input_length)                (batchSize, timeSteps, input_length+embedding_dim-1)
/// =======================================================================================================
/// </summary>
public sealed class EmbeddingLayer : Layer
{
    #region Private fields

    #region trainable parameters
    /// <summary>
    /// Word Embedding, of shape: (num_embeddings, embedding_dim)
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
    ///     num_embeddings:
    ///         Size of the vocabulary, i.e. maximum integer index + 1
    ///         In the input 'x' tensor:
    ///         each wordIndex element must be in [0, num_embeddings-1]
    ///     embedding_dim:
    ///         Dimension of the dense embedding
    ///     featureIndexInLastDimensionToUse:
    ///         index in last dimension of input tensor where to find the index of the feature to embed
    ///     embeddingTensorIndex:
    ///         index of the embedding tensor to use (in field 'EmbeddingTensors')
    /// </summary>
    private readonly List<(int num_embeddings, int embedding_dim, int indexInLastDimensionToUse, int embeddingTensorIndex)> EmbeddingDescriptions;

    private readonly List<(int num_embeddings, int embedding_dim)> EmbeddingTensorShapes;


    /// <summary>
    /// if value > 0 
    ///     clip values of weights gradients in range [-ClipValueForGradients, ClipValueForGradients]
    /// else
    ///     do not clip values
    /// </summary>
    private readonly float ClipValueForGradients;
    #endregion



    public static List<(int num_embeddings, int embedding_dim, int indexInLastDimensionToUse, int embeddingTensorIndex)> ToEmbeddingLayerDescription(
        int[] num_embeddings_array,
        int[] embedding_dim_array,
        int[] indexesInLastDimensionToUse, 
        int[] embeddingTensorIndex)
    {
        List<(int num_embeddings, int embedding_dim, int indexInLastDimensionToUse, int embeddingTensorIndex)> result = new();
        if (num_embeddings_array.Length != embedding_dim_array.Length || num_embeddings_array.Length != indexesInLastDimensionToUse.Length)
        {
            throw new ArgumentException($"input are not the same length : {num_embeddings_array.Length} vs {embedding_dim_array.Length} vs {indexesInLastDimensionToUse.Length}");
        }
        for (int i = 0; i < num_embeddings_array.Length; i++)
        {
            result.Add((num_embeddings_array[i], embedding_dim_array[i], indexesInLastDimensionToUse[i], embeddingTensorIndex[i]));
        }
        return result;
    }

    

    #region constructor
    public EmbeddingLayer(
        IEnumerable<(int num_embeddings, int embedding_dim, int indexInLastDimensionToUse, int embeddingTensorIndex)> embeddingDescriptions,
        float clipValueForGradients,
        bool trainable, Network network, string layerName) : base(network, layerName)
    {
        EmbeddingDescriptions = embeddingDescriptions.OrderBy(t => t.indexInLastDimensionToUse).ToList();
        EmbeddingTensorShapes = ExtractEmbeddingTensorShapes(EmbeddingDescriptions);

        if (EmbeddingDescriptions[0].indexInLastDimensionToUse < 0 && EmbeddingDescriptions.Count != 1)
        {
            throw new ArgumentException($"only 1 element is allowed if indexesInLastDimensionToUse = {EmbeddingDescriptions[0].indexInLastDimensionToUse}");
        }
        ClipValueForGradients = clipValueForGradients;
        
        Trainable = trainable;

        //trainable params
        int weightColumns = EmbeddingTensorShapes.Select(t=>t.num_embeddings*t.embedding_dim).Sum();
        _weights = GetFloatTensor(new[] { 1, weightColumns });
        _weightGradients = GetFloatTensor(_weights.Shape);

        _optimizer = Sample.GetOptimizer(_weights.Shape, null, MemoryPool);
        ResetParameters(false);
    }

    private static List<(int num_embeddings, int embedding_dim)> ExtractEmbeddingTensorShapes(List<(int num_embeddings, int embedding_dim, int indexInLastDimensionToUse, int embeddingTensorIndex)> embeddingDescriptions)
    {
        IDictionary<int, (int num_embeddings, int embedding_dim)> allEmbeddingTensors = new Dictionary<int, (int num_embeddings, int embedding_dim)>();
        foreach (var c in embeddingDescriptions)
        {
            if (!allEmbeddingTensors.ContainsKey(c.embeddingTensorIndex))
            {
                allEmbeddingTensors[c.embeddingTensorIndex] = (c.num_embeddings, c.embedding_dim);
            }
            else
            {
                var observedTensor = allEmbeddingTensors[c.embeddingTensorIndex];
                if (observedTensor.num_embeddings != c.num_embeddings || observedTensor.embedding_dim != c.embedding_dim)
                {
                    throw new ArgumentException($"embedding tensor {c.embeddingTensorIndex} has already been defined with different num_embeddings or embedding_dim");
                }
            }
        }
        return allEmbeddingTensors.OrderBy(t => t.Key).Select(t => t.Value).ToList();
    }

    #endregion

    #region forward and backward propagation
    public override void ForwardPropagation(List<Tensor> allX, Tensor y, bool isTraining)
    {
        Debug.Assert(allX.Count == 1);
        var x = allX[0];
        Debug.Assert(x.Shape[0] == y.Shape[0]); //same batchSize
        Debug.Assert(y.Shape.Length != 3 || x.Shape[1] == y.Shape[1]); //same timeSteps
        Debug.Assert(!ShouldEmbedEachElementOfLastDimension || x.Shape[1] == y.Shape[1]); //same timeSteps
        int deltaForIndexesInLastDimensionToUse = 0;
        var allEmbeddingTensors = Split(_weights);

        var xOriginalShape = (int[])x.Shape.Clone();
        var yOriginalShape = (int[])y.Shape.Clone();

        // we'll ensure that in all cases:
        //  the x shape is (batchSize, timeSteps, input_length)
        //  the y shape is (batchSize, timeSteps, input_length+embedding_dim-1)
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
                //y shape from (batchSize, input_length+embedding_dim-1) to (batchSize, 1, input_length+embedding_dim-1)
                y.ReshapeInPlace(new [] { y.Shape[0], 1, y.Shape[1] });
            }
        }

        if (ShouldEmbedEachElementOfLastDimension)
        {
            Debug.Assert(allEmbeddingTensors.Count == 1);
            y.WordEmbeddingForwardPropagation(x, allEmbeddingTensors[0], 0, 0, 0, 0);
        }
        else
        {
            for (var i = 0; i < EmbeddingDescriptions.Count; i++)
            {
                var embeddingTensor = allEmbeddingTensors[EmbeddingDescriptions[i].embeddingTensorIndex];
                var xIndexInLastDimensionToUse = EmbeddingDescriptions[i].indexInLastDimensionToUse;
                int copyCountBeforeIndex = (i == 0) ? xIndexInLastDimensionToUse : (xIndexInLastDimensionToUse - EmbeddingDescriptions[i-1].indexInLastDimensionToUse - 1);
                int copyCountAfterIndex = (i == EmbeddingDescriptions.Count - 1) ? x.Shape[2] - xIndexInLastDimensionToUse - 1 : 0;
                y.WordEmbeddingForwardPropagation(x, embeddingTensor, xIndexInLastDimensionToUse, deltaForIndexesInLastDimensionToUse + xIndexInLastDimensionToUse, copyCountBeforeIndex, copyCountAfterIndex);
                deltaForIndexesInLastDimensionToUse += embeddingTensor.Shape[1] - 1;
            }
        }

        x.ReshapeInPlace(xOriginalShape);
        y.ReshapeInPlace(yOriginalShape);
    }

    private List<Tensor> Split(Tensor w)
    {
        var res = new List<Tensor>();
        int nextIdxInWeights = 0;
        foreach(var (num_embeddings, embedding_dim) in EmbeddingTensorShapes)
        {
            var shape = new[] { num_embeddings, embedding_dim};
            res.Add(w.Slice(nextIdxInWeights, shape));
            nextIdxInWeights += shape[0] * shape[1];
        }
        return res;
    }
    private bool ShouldEmbedEachElementOfLastDimension => EmbeddingDescriptions[0].indexInLastDimensionToUse == -1;

    public override void BackwardPropagation(List<Tensor> allX, Tensor y_NotUsed, Tensor dy, List<Tensor> allDx)
    {
        Debug.Assert(y_NotUsed == null);
        Debug.Assert(allX.Count == 1);
        var x = allX[0];
        Debug.Assert(allDx.Count == 1);
        var dx = allDx[0]??GetFloatTensor(x.Shape);

        //we compute dW
        int deltaForIndexesInLastDimensionToUse = 0;
        var allEmbeddingTensorsGradients = Split(_weightGradients);

        


        var xOriginalShape = (int[])x.Shape.Clone();
        var dxOriginalShape = (int[])dx.Shape.Clone();
        var dyOriginalShape = (int[])dy.Shape.Clone();

        // we'll ensure that in all cases:
        //  the x shape is (batchSize, timeSteps, input_length)
        //  the y shape is (batchSize, timeSteps, input_length+embedding_dim-1)
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
                //dy shape from (batchSize, input_length+embedding_dim-1) to (batchSize, 1, input_length+embedding_dim-1)
                dy.ReshapeInPlace(new[] { dy.Shape[0], 1, dy.Shape[1] });
            }
        }

        if (ShouldEmbedEachElementOfLastDimension)
        {
            Debug.Assert(allEmbeddingTensorsGradients.Count == 1);
            allEmbeddingTensorsGradients[0].WordEmbeddingBackwardPropagation(x, dx, dy, 0, 0, 0, 0);
        }
        else
        {
            for (var i = 0; i < EmbeddingDescriptions.Count; i++)
            {
                var embeddingTensorsGradients = allEmbeddingTensorsGradients[EmbeddingDescriptions[i].embeddingTensorIndex];
                var dxIndexInLastDimensionToUse = EmbeddingDescriptions[i].indexInLastDimensionToUse;
                int copyCountBeforeIndex = (i == 0) ? dxIndexInLastDimensionToUse : (dxIndexInLastDimensionToUse - EmbeddingDescriptions[i - 1].indexInLastDimensionToUse - 1);
                int copyCountAfterIndex = (i == EmbeddingDescriptions.Count - 1) ? dx.Shape[2] - dxIndexInLastDimensionToUse - 1 : 0;
                embeddingTensorsGradients.WordEmbeddingBackwardPropagation(x, dx, dy, dxIndexInLastDimensionToUse, deltaForIndexesInLastDimensionToUse + dxIndexInLastDimensionToUse, copyCountBeforeIndex, copyCountAfterIndex);
                deltaForIndexesInLastDimensionToUse += embeddingTensorsGradients.Shape[1] - 1;
            }
        }

        x.ReshapeInPlace(xOriginalShape);
        dx.ReshapeInPlace(dxOriginalShape);
        dy.ReshapeInPlace(dyOriginalShape);

        if (ClipValueForGradients > 1e-6)
        {
            _weightGradients.Clip(-ClipValueForGradients, ClipValueForGradients);
        }

        //weight_decay on dW
        if (Sample.Use_weight_decay_in_backpropagation)
        {
            int batchSize = dy.Shape[0];
            var alpha = 2 * batchSize * (float)Sample.weight_decay;
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
            .Add(nameof(embedding_dim_array), embedding_dim_array)
            .Add(nameof(IndexesInLastDimensionToUse), IndexesInLastDimensionToUse)
            .Add(nameof(EmbeddingTensorIndex), EmbeddingTensorIndex)
            .Add(nameof(ClipValueForGradients), ClipValueForGradients)
            .ToString();
    }

    public int[] VocabularySizes => EmbeddingDescriptions.Select(t => t.num_embeddings).ToArray();
    public int[] embedding_dim_array => EmbeddingDescriptions.Select(t => t.embedding_dim).ToArray();
    public int[] IndexesInLastDimensionToUse => EmbeddingDescriptions.Select(t => t.indexInLastDimensionToUse).ToArray();
    public int[] EmbeddingTensorIndex => EmbeddingDescriptions.Select(t => t.embeddingTensorIndex).ToArray();

    public static EmbeddingLayer Deserialize(IDictionary<string, object> serialized, Network network)
    {
        int[] VocabularySizes = serialized.ContainsKey("num_embeddings")
            ? new[] { (int)serialized["num_embeddings"] }
            : (int[])serialized[nameof(VocabularySizes)];
        int[] embedding_dim_array = serialized.ContainsKey("embedding_dim")
            ? new[] { (int)serialized["embedding_dim"] }
            : (int[])serialized[nameof(embedding_dim_array)];
        int[] IndexesInLastDimensionToUse = serialized.ContainsKey("IndexInLastDimensionToUse")
            ? new[] { (int)serialized["IndexInLastDimensionToUse"] }
            : (int[])serialized[nameof(IndexesInLastDimensionToUse)];
        int[] EmbeddingTensorIndex = serialized.ContainsKey("EmbeddingTensorIndex")
            ? new[] { (int)serialized["EmbeddingTensorIndex"] }
            : (int[])serialized[nameof(EmbeddingTensorIndex)];

        return new EmbeddingLayer(
            ToEmbeddingLayerDescription(VocabularySizes, embedding_dim_array, IndexesInLastDimensionToUse, EmbeddingTensorIndex),
            (float)serialized[nameof(ClipValueForGradients)],
            (bool)serialized[nameof(Trainable)],
            network,
            (string)serialized[nameof(LayerName)]);
    }
    public override void AddToOtherNetwork(Network otherNetwork) { AddToOtherNetwork(otherNetwork, Deserialize); }
    #endregion

    #region PyTorch support
    //see : https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    public override void ToPytorchModule(List<string> constructorLines, List<string> forwardLines)
    {
        var line = "self." + LayerName + " = torch.nn.Embedding(num_embeddings=" + EmbeddingTensorShapes[0].num_embeddings + ", embedding_dim=" + EmbeddingTensorShapes[0].embedding_dim + ")";
        if (ClipValueForGradients>1e-6)
        {
            line += $" # ClipValueForGradients={ClipValueForGradients} is not supported in PyTorch";
        }
        if (EmbeddingTensorShapes.Count != 1)
        {
            line += $" # EmbeddingTensorShapes.Count={EmbeddingTensorShapes.Count} is not supported in PyTorch";
        }
        constructorLines.Add(line);
        UpdateForwardLines(forwardLines);
    }

    #endregion

    public override int[] OutputShape(int batchSize)
    {
        var outputShape = Utils.CloneShapeWithNewCount(PrevLayer.OutputShape(batchSize), batchSize);
        if (ShouldEmbedEachElementOfLastDimension)
        {
            Debug.Assert(IndexesInLastDimensionToUse.Length == 1);
            //Debug.Assert(prevLayerOutputShape.Length == 2);
            outputShape = outputShape.Append(embedding_dim_array[0]).ToArray();
            return outputShape;
        }
        else
        {
            //Debug.Assert(prevLayerOutputShape.Length == 3);
            outputShape[^1] += embedding_dim_array.Sum() - embedding_dim_array.Length;
            return outputShape;
        }
    }
    public override string ToString()
    {
        var result = LayerName + ": " + ShapeChangeDescription();
        result += " " + _weights + " (" + TotalParams + " neurons)";
        return result;
    }
}