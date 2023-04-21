using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using JetBrains.Annotations;
using SharpNet.Data;
using SharpNet.Networks;
using SharpNet.Optimizers;

namespace SharpNet.Layers;
/// <summary>
/// Input:
///  allX[0] = Q of shape (batch_size, query_timeSteps == input_seq_length, embedding_dim)
///  allX[1] = V of shape (batch_size, value_timeSteps, embedding_dim)
///  allX[2] = K of shape (batch_size, value_timeSteps, embedding_dim)
///     in most cases, K and V are the same tensor (K is optional, it's default value is V)
/// Output:
/// y of shape:           (batch_size, value_timeSteps, embedding_dim)
///                       (same as V shape)
/// </summary>
public class MultiHeadAttentionLayer : Layer
{
    private readonly int _num_heads;
    private readonly int _key_dim;
    private readonly int _value_dim;
    private readonly bool _use_bias_Q_V_K;
    private readonly bool _use_bias_O;
    private readonly bool _use_causal_mask;
    private const bool use_scale = true;
    private const bool flattenInputTensorOnLastDimension = true;
    private const int QUERIES_LAYER_INDEX = 0;
    private const int VALUES_LAYER_INDEX = 1;
    private const int KEYS_LAYER_INDEX = 2;


    #region Private fields
    #region trainable parameters
    [NotNull] private readonly Tensor _weights;
    [CanBeNull] private readonly Tensor _bias;
    #endregion
    #region gradients
    [NotNull] private readonly Tensor _weightGradients;
    [CanBeNull] private readonly Tensor _biasGradients;
    #endregion
    #endregion

    #region trainable parameters
    /// <summary>
    /// shape :   (embedding_dim, 2*num_heads * key_dim+2*num_heads* value_dim)
    /// contains the Weights for Q, K,V & O
    /// </summary>
    public override Tensor Weights => _weights;
    /// <summary>
    /// shape :   (1, 2*num_heads*key_dim + num_heads* value_dim + embedding_dim)
    /// contains the Weights for Q, K,V & O
    /// </summary>
    public override Tensor Bias => _bias;
    public override Tensor WeightGradients => _weightGradients;
    public override Tensor BiasGradients => _biasGradients;


    #endregion

    [NotNull] private readonly Optimizer _w_Q_optimizer;
    [NotNull] private readonly Optimizer _w_K_optimizer;
    [NotNull] private readonly Optimizer _w_V_optimizer;
    [NotNull] private readonly Optimizer _w_O_optimizer;


    //private Tensor Q_heads;             //(batch_size*query_time_steps, _num_heads*_key_dim)
    //private Tensor K_heads;             //(batch_size*value_time_steps, _num_heads*_key_dim)
    //private Tensor V_heads;             //(batch_size*value_time_steps, _num_heads, _value_dim)
    //private Tensor attention_heads;     //(batch_size*_num_heads, value_time_steps, _value_dim)
    private Tensor weights_buffer;      //(batch_size, query_time_steps, value_time_steps)
    private Tensor Q_heads_T;           //(batch_size*_num_heads, query_time_steps, _key_dim)
    private Tensor K_heads_T;           //(batch_size*_num_heads, value_time_steps, _key_dim)
    private Tensor V_heads_T;           //(batch_size*_num_heads, value_time_steps, _value_dim)
    private Tensor attention_heads_T;   //(batch_size, value_time_steps, _num_heads* _value_dim)


    /// <summary>
    /// shape of Weights for Q K V O tensors
    /// </summary>
    private readonly List<int[]> _shapes_w_Q_K_V_O;
    /// <summary>
    /// number of elements in Weights for Q K V O tensors
    /// </summary>
    private readonly List<int> _count_w_Q_K_V_O;

    /// <summary>
    /// shape of Weights Bias for Q K V O tensors
    /// </summary>
    private readonly List<int[]> _shapes_w_bias_Q_K_V_O;
    /// <summary>
    /// number of elements in Weights Bias for Q K V O tensors
    /// </summary>
    private readonly List<int> _count_w_bias_Q_K_V_O;

    /// <summary>
    /// no need to have 'embedding_dim' as a parameter: it is always equal to the last dimension of 'V' (value) Layer
    /// </summary>
    /// <param name="num_heads"></param>
    /// <param name="key_dim"></param>
    /// <param name="value_dim"></param>
    /// <param name="use_bias_Q_V_K"></param>
    /// <param name="use_bias_O"></param>
    /// <param name="use_causal_mask"></param>
    /// <param name="queriesLayerIndex"></param>
    /// <param name="valuesLayerIndex"></param>
    /// <param name="keysLayerIndex"></param>
    /// <param name="network"></param>
    /// <param name="layerName"></param>
    public MultiHeadAttentionLayer(int num_heads, int key_dim, int value_dim, bool use_bias_Q_V_K, bool use_bias_O,
        bool use_causal_mask, int queriesLayerIndex, int valuesLayerIndex, int keysLayerIndex,
        Network network, string layerName = "") : base(network,
        new[] { queriesLayerIndex, valuesLayerIndex, keysLayerIndex }, layerName)
    {
        _num_heads = num_heads;
        _key_dim = key_dim;
        _value_dim = value_dim;
        _use_bias_Q_V_K = use_bias_Q_V_K;
        _use_bias_O = use_bias_O;
        var embedding_dim = network.Layers[valuesLayerIndex].OutputShape(1)[2];
        _use_causal_mask = use_causal_mask;

        _shapes_w_Q_K_V_O = new List<int[]>
        {
            new[] { embedding_dim, num_heads * key_dim },
            new[] { embedding_dim, num_heads * key_dim },
            new[] { embedding_dim, num_heads * value_dim },
            new[] { num_heads * value_dim, embedding_dim }
        };
        _count_w_Q_K_V_O = _shapes_w_Q_K_V_O.Select(Utils.Product).ToList();

        _shapes_w_bias_Q_K_V_O =new List<int[]>
            {
                _use_bias_Q_V_K? new[] { 1, num_heads * key_dim }:null,
                _use_bias_Q_V_K? new[] { 1, num_heads * key_dim }:null,
                _use_bias_Q_V_K? new[] { 1, num_heads * value_dim }:null,
                _use_bias_O? new[] { 1, embedding_dim }:null,
            };
        _count_w_bias_Q_K_V_O = _shapes_w_bias_Q_K_V_O.Select(s => s==null?0:Utils.Product(s)).ToList();

        //trainable params
        _weights = GetFloatTensor(new[] { embedding_dim, 2 * num_heads * key_dim + 2 * num_heads * value_dim });
        _weightGradients = GetFloatTensor(_weights.Shape);
        _bias = _count_w_bias_Q_K_V_O.Sum() > 0 ? GetFloatTensor(new[] { 1, _count_w_bias_Q_K_V_O.Sum() }) : null;
        _biasGradients = _bias == null ? null:GetFloatTensor(_bias.Shape);

        _w_Q_optimizer = Sample.GetOptimizer(_shapes_w_Q_K_V_O[0], _shapes_w_bias_Q_K_V_O[0], MemoryPool);
        _w_K_optimizer = Sample.GetOptimizer(_shapes_w_Q_K_V_O[1], _shapes_w_bias_Q_K_V_O[1], MemoryPool);
        _w_V_optimizer = Sample.GetOptimizer(_shapes_w_Q_K_V_O[2], _shapes_w_bias_Q_K_V_O[2], MemoryPool);
        _w_O_optimizer = Sample.GetOptimizer(_shapes_w_Q_K_V_O[3], _shapes_w_bias_Q_K_V_O[3], MemoryPool);

        // ReSharper disable once VirtualMemberCallInConstructor
        ResetParameters(false);
    }


    public override void UpdateWeights(int batchSize, double learningRate, double maxLearningRate)
    {
        Debug.Assert(Network.IsMaster);
        if (Trainable)
        {
            _w_Q_optimizer.UpdateWeights(learningRate, maxLearningRate, batchSize, w_Q, w_Q_Gradients, w_Q_bias, w_Q_bias_Gradients);
            _w_K_optimizer.UpdateWeights(learningRate, maxLearningRate, batchSize, w_K, w_K_Gradients, w_K_bias, w_K_bias_Gradients);
            _w_V_optimizer.UpdateWeights(learningRate, maxLearningRate, batchSize, w_V, w_V_Gradients, w_V_bias, w_V_bias_Gradients);
            _w_O_optimizer.UpdateWeights(learningRate, maxLearningRate, batchSize, w_O, w_O_Gradients, w_O_bias, w_O_bias_Gradients);
        }
    }

    #region forward and backward propagation

    public override void ForwardPropagation(List<Tensor> allX, Tensor y, bool isTraining)
    {
        Debug.Assert(allX.Count == 3);

        var Q = allX[QUERIES_LAYER_INDEX]; // queries: (batch_size, query_timeSteps == input_seq_length, embedding_dim)
        var V = allX[VALUES_LAYER_INDEX]; // values:  (batch_size, value_timeSteps, embedding_dim)
        var K = allX[KEYS_LAYER_INDEX]; // keys:    (batch_size, value_timeSteps, embedding_dim)

        if (!V.Shape.SequenceEqual(K.Shape))
        {
            throw new ArgumentException($"V.Shape and K.Shape must be equal, but are {Tensor.ShapeToString(V.Shape)} and {Tensor.ShapeToString(K.Shape)}");
        }
        if (!V.Shape.SequenceEqual(y.Shape))
        {
            throw new ArgumentException($"V.Shape and y.Shape must be equal, but are {Tensor.ShapeToString(V.Shape)} and {Tensor.ShapeToString(y.Shape)}");
        }
        if (Q.Shape[2] != V.Shape[2])
        {
            throw new ArgumentException($"queries.Shape[2] and values.Shape[2] must be equal (same embedding dim), but are {Q.Shape[2]} and {V.Shape[2]}");
        }

        var batch_size = K.Shape[0];
        var query_time_steps = Q.Shape[1];
        var key_value_time_steps = V.Shape[1];

        var Q_K_V_heads_buffer = GetFloatTensor(new[] {Utils.Max(batch_size * query_time_steps*_num_heads * _key_dim, batch_size * key_value_time_steps, _num_heads * _key_dim, batch_size * key_value_time_steps*_num_heads * _value_dim) });

        var Q_heads = Q_K_V_heads_buffer.Reshape(batch_size*query_time_steps, _num_heads*_key_dim);
        DenseLayer.DenseForwardPropagation(Q_heads, Q, w_Q, w_Q_bias, flattenInputTensorOnLastDimension);
        GetFloatTensor(ref Q_heads_T , new[] { batch_size, _num_heads, query_time_steps, _key_dim });
        Q_heads.Reshape(batch_size, query_time_steps, _num_heads, _key_dim).TransposeSecondAndThirdDimension(Q_heads_T);
        Q_heads_T.ReshapeInPlace(batch_size*_num_heads, query_time_steps, _key_dim);

        var K_heads = Q_K_V_heads_buffer.Reshape(batch_size * key_value_time_steps, _num_heads * _key_dim);
        DenseLayer.DenseForwardPropagation(K_heads, K, w_K, w_K_bias, flattenInputTensorOnLastDimension);
        GetFloatTensor(ref K_heads_T, new[] { batch_size, _num_heads, key_value_time_steps, _key_dim });
        K_heads.Reshape(batch_size, key_value_time_steps, _num_heads, _key_dim).TransposeSecondAndThirdDimension(K_heads_T);
        K_heads_T.ReshapeInPlace(batch_size * _num_heads, key_value_time_steps, _key_dim);

        var V_heads = Q_K_V_heads_buffer.Reshape(batch_size * key_value_time_steps, _num_heads * _value_dim);
        DenseLayer.DenseForwardPropagation(V_heads, V, w_V, w_V_bias, flattenInputTensorOnLastDimension);
        GetFloatTensor(ref V_heads_T, new[] { batch_size, _num_heads, key_value_time_steps, _value_dim });
        V_heads.Reshape(batch_size, key_value_time_steps, _num_heads, _value_dim).TransposeSecondAndThirdDimension(V_heads_T);
        V_heads_T.ReshapeInPlace(batch_size * _num_heads, key_value_time_steps, _value_dim);

        var attention_heads = Q_K_V_heads_buffer.Reshape(V_heads_T.Shape);
        ScaledDotProductAttentionLayer.ScaledDotProductAttentionForwardPropagation(Q_heads_T, V_heads_T, K_heads_T, attention_heads, isTraining, ref weights_buffer, Network.MemoryPool, use_scale, _use_causal_mask);

        GetFloatTensor(ref attention_heads_T, new[] { batch_size, key_value_time_steps, _num_heads, _value_dim });
        attention_heads.Reshape(batch_size, _num_heads, key_value_time_steps, _value_dim).TransposeSecondAndThirdDimension(attention_heads_T);
        attention_heads_T.ReshapeInPlace(batch_size, key_value_time_steps, _num_heads* _value_dim);
        DenseLayer.DenseForwardPropagation(y, attention_heads_T, w_O, w_O_bias, flattenInputTensorOnLastDimension);

        FreeFloatTensor(Q_K_V_heads_buffer);
        if (!isTraining)
        {
            FreeFloatTensor(ref weights_buffer);
            FreeFloatTensor(ref Q_heads_T);
            FreeFloatTensor(ref K_heads_T);
            FreeFloatTensor(ref V_heads_T);
            FreeFloatTensor(ref attention_heads_T);
        }
    }

    public override void BackwardPropagation(List<Tensor> allX, Tensor y_NotUsed, Tensor dy, List<Tensor> allDx)
    {
        ////dy:          (batch_size, value_timeSteps, value_embedding_dim)
        var dQ = allDx[QUERIES_LAYER_INDEX];    // queries:    (batch_size, query_timeSteps == input_seq_length, embedding_dim)
        var dV = allDx[VALUES_LAYER_INDEX];     // values:     (batch_size, value_timeSteps, embedding_dim)
        var dK = allDx[KEYS_LAYER_INDEX];       // keys:       (batch_size, value_timeSteps, embedding_dim)
        //Debug.Assert(weights_buffer != null);
        var batch_size = dK.Shape[0];
        var query_time_steps = dQ.Shape[1];
        var value_time_steps = dV.Shape[1];
        var Q = allX[QUERIES_LAYER_INDEX];
        var V = allX[VALUES_LAYER_INDEX];
        var K = allX[KEYS_LAYER_INDEX];
        
        var dQ_dK_dV_heads_buffer = GetFloatTensor(new[] { Math.Max(batch_size * query_time_steps * _num_heads * _key_dim, batch_size * value_time_steps * _num_heads * _value_dim) });

        var d_attention_heads_T = dQ_dK_dV_heads_buffer.Reshape(attention_heads_T.Shape);
        DenseLayer.DenseBackwardPropagation(d_attention_heads_T, w_O_Gradients, w_O_bias_Gradients,
            attention_heads_T, dy, w_O,
            Network.Sample, 0, false, flattenInputTensorOnLastDimension);

        var d_attention_heads = GetFloatTensor(V_heads_T.Shape);
        d_attention_heads_T.Reshape(batch_size, value_time_steps, _num_heads, _value_dim).TransposeSecondAndThirdDimension(d_attention_heads);
        d_attention_heads.ReshapeInPlace(batch_size * _num_heads, value_time_steps, _value_dim);

        var dQ_heads_T = GetFloatTensor(Q_heads_T.Shape);
        var dK_heads_T = GetFloatTensor(K_heads_T.Shape);
        var dV_heads_T = GetFloatTensor(V_heads_T.Shape);

        ScaledDotProductAttentionLayer.ScaledDotProductAttentionBackwardPropagation( 
            /* Out */ dQ_heads_T, /* Out */dV_heads_T, /* Out */ dK_heads_T, 
            /* In */ Q_heads_T, /* In */ V_heads_T, /* In */ K_heads_T, /* In */ d_attention_heads, /* In */ ref weights_buffer,
            Network.MemoryPool, use_scale);

        dQ_heads_T.Reshape(batch_size, _num_heads, query_time_steps, _key_dim).TransposeSecondAndThirdDimension(dQ_dK_dV_heads_buffer  /* dQ_heads */);
        DenseLayer.DenseBackwardPropagation(dQ, w_Q_Gradients, w_Q_bias_Gradients, Q, dQ_dK_dV_heads_buffer.Reshape(batch_size * query_time_steps, -1), w_Q, Network.Sample, 0, false, flattenInputTensorOnLastDimension);

        dK_heads_T.Reshape(batch_size, _num_heads, query_time_steps, _key_dim).TransposeSecondAndThirdDimension(dQ_dK_dV_heads_buffer  /* dK_heads */);
        DenseLayer.DenseBackwardPropagation(dK, w_K_Gradients, w_K_bias_Gradients, K, dQ_dK_dV_heads_buffer.Reshape(batch_size * query_time_steps, -1), w_K, Network.Sample, 0, false, flattenInputTensorOnLastDimension);

        dV_heads_T.Reshape(batch_size, _num_heads, value_time_steps, _value_dim).TransposeSecondAndThirdDimension(dQ_dK_dV_heads_buffer /* dV_heads */);
        DenseLayer.DenseBackwardPropagation(dV, w_V_Gradients, w_V_bias_Gradients, V, dQ_dK_dV_heads_buffer.Reshape(batch_size * value_time_steps, -1), w_V, Network.Sample, 0, false, flattenInputTensorOnLastDimension);

        FreeFloatTensor(dQ_dK_dV_heads_buffer);
        FreeFloatTensor(d_attention_heads);
        FreeFloatTensor(dQ_heads_T);
        FreeFloatTensor(dK_heads_T);
        FreeFloatTensor(dV_heads_T);
        FreeFloatTensor(ref weights_buffer);
        FreeFloatTensor(ref Q_heads_T);
        FreeFloatTensor(ref K_heads_T);
        FreeFloatTensor(ref V_heads_T);
        FreeFloatTensor(ref attention_heads_T);
    }

    

    public override string LayerType() { return "multi_head_attention"; }
    private Optimizer[] AllOptimizer => new[] { _w_Q_optimizer, _w_K_optimizer, _w_V_optimizer, _w_O_optimizer };


    public override void ResetParameters(bool resetAlsoOptimizerWeights = true)
    {
        //trainable params
        foreach (var w in new[] { w_Q, w_K, w_V, w_O })
        {
            w.GlorotUniform(Rand);
        }
        _bias?.ZeroMemory();
        if (resetAlsoOptimizerWeights)
        {
            Array.ForEach(AllOptimizer, o => o.ZeroMemory());
        }
    }

    public override bool OutputNeededForBackwardPropagation => false;
    public override bool InputNeededForBackwardPropagation => true;
    #endregion

    #region serialization

    public override string Serialize()
    {
        return RootSerializer()
            .Add(nameof(_num_heads), _num_heads)
            .Add(nameof(_key_dim), _key_dim)
            .Add(nameof(_value_dim), _value_dim)
            .Add(nameof(_use_bias_Q_V_K), _use_bias_Q_V_K)
            .Add(nameof(_use_bias_O), _use_bias_O)
            .Add(nameof(_use_causal_mask), _use_causal_mask)
            .ToString();
    }
    public static MultiHeadAttentionLayer Deserialize(IDictionary<string, object> serialized, Network network)
    {
        var num_heads = (int)serialized[nameof(_num_heads)];
        var key_dim = (int)serialized[nameof(_key_dim)];
        var value_dim = (int)serialized[nameof(_value_dim)];
        var use_bias_Q_V_K = (bool)serialized[nameof(_use_bias_Q_V_K)];
        var use_bias_O = (bool)serialized[nameof(_use_bias_O)];
        var use_causal_mask = (bool)serialized[nameof(_use_causal_mask)];
        var previousLayerIndexes = (int[])serialized[nameof(PreviousLayerIndexes)];
        return new MultiHeadAttentionLayer(num_heads, key_dim, value_dim, use_bias_Q_V_K, use_bias_O, use_causal_mask, previousLayerIndexes[0], previousLayerIndexes[1], previousLayerIndexes[2], network, (string)serialized[nameof(LayerName)]);
    }
    public override void AddToOtherNetwork(Network otherNetwork) { AddToOtherNetwork(otherNetwork, Deserialize); }
    #endregion

    private string WeightDatasetPath => DatasetNameToDatasetPath("kernel:0");
    private string BiasDatasetPath => DatasetNameToDatasetPath("bias:0");

    public override List<Tuple<Tensor, string>> Parameters
    {
        get
        {
            var result = new List<Tuple<Tensor, string>>
            {
                Tuple.Create(_weights, WeightDatasetPath),
                Tuple.Create(_bias, BiasDatasetPath)
            };
            result.RemoveAll(t => t.Item1 == null);
            return result;
        }
    }
    protected override List<Tensor> EmbeddedTensors(bool includeOptimizeTensors)
    {
        var result = Parameters.Select(t => t.Item1).Concat(ParameterGradients).ToList();
        result.AddRange(new[] { weights_buffer, Q_heads_T, K_heads_T, V_heads_T,  });
        if (includeOptimizeTensors)
        {
            Array.ForEach(AllOptimizer, o => result.AddRange(o.EmbeddedTensors));
        }
        result.RemoveAll(t => t == null);
        return result;
    }

    protected override Optimizer Optimizer => throw new ArgumentException("should never be called");

    public override void Dispose()
    {
        if (_isDisposed)
        {
            return;
        }
        _isDisposed = true;
        EmbeddedTensors(false).ForEach(FreeFloatTensor);
        Array.ForEach(AllOptimizer, o => o?.Dispose());
    }

    /// <summary>
    /// the output shape of the layer is the same as the input 'V' layer
    /// </summary>
    /// <param name="batchSize"></param>
    /// <returns></returns>
    public override int[] OutputShape(int batchSize)
    {
        return PreviousLayers[VALUES_LAYER_INDEX].OutputShape(batchSize);
    }

    public Tensor w_Q => _weights.Slice(0, _shapes_w_Q_K_V_O[0]); // (embedding_dim, num_heads*key_dim)
    public Tensor w_K => _weights.Slice(_count_w_Q_K_V_O[0], _shapes_w_Q_K_V_O[1]); // (embedding_dim, num_heads*key_dim)
    public Tensor w_V => _weights.Slice(_count_w_Q_K_V_O[0] + _count_w_Q_K_V_O[1], _shapes_w_Q_K_V_O[2]); // (embedding_dim, num_heads*value_dim)
    public Tensor w_O => _weights.Slice(_count_w_Q_K_V_O[0] + _count_w_Q_K_V_O[1] + _count_w_Q_K_V_O[2], _shapes_w_Q_K_V_O[3]); // (num_heads*value_dim, embedding_dim)

    private Tensor w_Q_bias => !_use_bias_Q_V_K ? null:_bias?.Slice(0, _shapes_w_bias_Q_K_V_O[0]); // (1, num_heads*key_dim)
    private Tensor w_K_bias => !_use_bias_Q_V_K ? null : _bias?.Slice(_count_w_bias_Q_K_V_O[0], _shapes_w_bias_Q_K_V_O[1]); // (1, num_heads*key_dim)
    private Tensor w_V_bias => !_use_bias_Q_V_K ? null : _bias?.Slice(_count_w_bias_Q_K_V_O[0] + _count_w_bias_Q_K_V_O[1], _shapes_w_bias_Q_K_V_O[2]); // (1, num_heads*value_dim)
    private Tensor w_O_bias => !_use_bias_O ?null: _bias?.Slice(_count_w_bias_Q_K_V_O[0] + _count_w_bias_Q_K_V_O[1] + _count_w_bias_Q_K_V_O[2], _shapes_w_bias_Q_K_V_O[3]); // (1, embedding_dim)
    private Tensor w_Q_Gradients => _weightGradients.Slice(0, _shapes_w_Q_K_V_O[0]); // (embedding_dim, num_heads*key_dim)
    private Tensor w_K_Gradients => _weightGradients.Slice(_count_w_Q_K_V_O[0], _shapes_w_Q_K_V_O[1]); // (embedding_dim, num_heads*key_dim)
    private Tensor w_V_Gradients => _weightGradients.Slice(_count_w_Q_K_V_O[0] + _count_w_Q_K_V_O[1], _shapes_w_Q_K_V_O[2]); // (embedding_dim, num_heads*value_dim)
    private Tensor w_O_Gradients => _weightGradients.Slice(_count_w_Q_K_V_O[0] + _count_w_Q_K_V_O[1] + _count_w_Q_K_V_O[2], _shapes_w_Q_K_V_O[3]); // (num_heads*value_dim, embedding_dim)
    private Tensor w_Q_bias_Gradients => !_use_bias_Q_V_K ? null : _biasGradients?.Slice(0, _shapes_w_bias_Q_K_V_O[0]); // (1, num_heads*key_dim)
    private Tensor w_K_bias_Gradients => !_use_bias_Q_V_K ? null : _biasGradients?.Slice(_count_w_bias_Q_K_V_O[0], _shapes_w_bias_Q_K_V_O[1]); // (1, num_heads*key_dim)
    private Tensor w_V_bias_Gradients => !_use_bias_Q_V_K ? null : _biasGradients?.Slice(_count_w_bias_Q_K_V_O[0] + _count_w_bias_Q_K_V_O[1], _shapes_w_bias_Q_K_V_O[2]); // (1, num_heads*value_dim)
    private Tensor w_O_bias_Gradients => !_use_bias_O?null: _biasGradients?.Slice(_count_w_bias_Q_K_V_O[0] + _count_w_bias_Q_K_V_O[1] + _count_w_bias_Q_K_V_O[2], _shapes_w_bias_Q_K_V_O[3]); // (1, embedding_dim)


}