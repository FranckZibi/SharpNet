using System;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Security.Policy;
using JetBrains.Annotations;
using ProtoBuf.Meta;
using SharpNet.Data;
using SharpNet.Networks;
using SharpNet.Optimizers;

namespace SharpNet.Layers;
/// <summary>
/// Input:
///  allX[0] = Q of shape (batch_size, query_timeSteps == input_seq_length, embed_dim)
///  allX[1] = V of shape (batch_size, value_timeSteps, embed_dim)
///  allX[2] = K of shape (batch_size, value_timeSteps, embed_dim)
///     in most cases, K and V are the same tensor (K is optional, it's default value is V)
/// Output:
/// y of shape:           (batch_size, value_timeSteps, embed_dim)
///                       (same as V shape)
/// </summary>
public class MultiheadAttention : Layer
{
    private readonly int _num_heads;
    private readonly int _key_dim;
    private readonly int _value_dim;
    private readonly bool _use_bias_Q_K_V;
    private readonly bool _use_bias_O;
    private readonly bool _is_causal;
    private const bool use_scale = true;
    private const bool flattenInputTensorOnLastDimension = true;
    private const int QUERY_LAYER_INDEX = 0;
    private const int KEY_LAYER_INDEX = 1;
    private const int VALUE_LAYER_INDEX = 2;


    #region Private fields
    #region trainable parameters
    [NotNull] private Tensor _weights;
    [CanBeNull] private Tensor _bias;
    #endregion
    #region gradients
    [NotNull] private Tensor _weightGradients;
    [CanBeNull] private Tensor _biasGradients;
    #endregion
    #endregion

    #region trainable parameters
    /// <summary>
    /// shape :   (embed_dim, 2*num_heads * key_dim+2*num_heads* value_dim)
    /// contains the Weights for Q, K,V & O
    /// </summary>
    public override Tensor Weights => _weights;
    /// <summary>
    /// shape :   (2*num_heads*key_dim + num_heads* value_dim + embed_dim)
    /// contains the Weights for Q, K,V & O
    /// </summary>
    public override Tensor Bias => _bias;
    public override Tensor WeightGradients => _weightGradients;
    public override Tensor BiasGradients => _biasGradients;


    #endregion

    [NotNull] private readonly Optimizer _w_Q_optimizer;
    [NotNull] private readonly Optimizer _w_K_optimizer;
    [NotNull] private readonly Optimizer _w_V_optimizer;
    [NotNull] private readonly Optimizer _out_proj_weight_optimizer;


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
    /// no need to have 'embed_dim' as a parameter: it is always equal to the last dimension of 'V' (value) Layer
    /// </summary>
    /// <param name="num_heads"></param>
    /// <param name="key_dim"></param>
    /// <param name="value_dim"></param>
    /// <param name="use_bias_Q_K_V"></param>
    /// <param name="use_bias_O"></param>
    /// <param name="is_causal"></param>
    /// <param name="queriesLayerIndex"></param>
    /// <param name="keysLayerIndex"></param>
    /// <param name="valuesLayerIndex"></param>
    /// <param name="network"></param>
    /// <param name="layerName"></param>
    public MultiheadAttention(int num_heads, int key_dim, int value_dim, bool use_bias_Q_K_V, bool use_bias_O,
        bool is_causal, int queriesLayerIndex, int keysLayerIndex, int valuesLayerIndex,
        Network network, string layerName = "") : base(network,
        new[] { queriesLayerIndex, keysLayerIndex, valuesLayerIndex }, layerName)
    {
        _num_heads = num_heads;
        _key_dim = key_dim;
        _value_dim = value_dim;
        _use_bias_Q_K_V = use_bias_Q_K_V;
        _use_bias_O = use_bias_O;
        var embed_dim = network.Layers[valuesLayerIndex].OutputShape(1)[2];
        _is_causal = is_causal;

        _shapes_w_Q_K_V_O = new List<int[]>
        {
            new[] { num_heads * key_dim, embed_dim },
            new[] { num_heads * key_dim, embed_dim },
            new[] { num_heads * value_dim, embed_dim },
            new[] { num_heads * value_dim, embed_dim }
        };
        _count_w_Q_K_V_O = _shapes_w_Q_K_V_O.Select(Utils.Product).ToList();

        _shapes_w_bias_Q_K_V_O =new List<int[]>
            {
                _use_bias_Q_K_V? new[] { _shapes_w_Q_K_V_O[0][1] }:null,
                _use_bias_Q_K_V? new[] { _shapes_w_Q_K_V_O[1][1] }:null,
                _use_bias_Q_K_V? new[] { _shapes_w_Q_K_V_O[2][1] }:null,
                _use_bias_O? new[] { embed_dim }:null,
            };
        _count_w_bias_Q_K_V_O = _shapes_w_bias_Q_K_V_O.Select(s => s==null?0:Utils.Product(s)).ToList();

        //trainable params
        _weights = GetFloatTensor(new[] { 2 * num_heads * key_dim + 2 * num_heads * value_dim, embed_dim });
        _weightGradients = GetFloatTensor(_weights.Shape);
        _bias = _count_w_bias_Q_K_V_O.Sum() > 0 ? GetFloatTensor(new[] { _count_w_bias_Q_K_V_O.Sum() }) : null;
        _biasGradients = _bias == null ? null:GetFloatTensor(_bias.Shape);

        _w_Q_optimizer = Sample.GetOptimizer(_shapes_w_Q_K_V_O[0], _shapes_w_bias_Q_K_V_O[0], MemoryPool);
        _w_K_optimizer = Sample.GetOptimizer(_shapes_w_Q_K_V_O[1], _shapes_w_bias_Q_K_V_O[1], MemoryPool);
        _w_V_optimizer = Sample.GetOptimizer(_shapes_w_Q_K_V_O[2], _shapes_w_bias_Q_K_V_O[2], MemoryPool);
        _out_proj_weight_optimizer = Sample.GetOptimizer(_shapes_w_Q_K_V_O[3], _shapes_w_bias_Q_K_V_O[3], MemoryPool);

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
            _out_proj_weight_optimizer.UpdateWeights(learningRate, maxLearningRate, batchSize, out_proj_weight, out_proj_weight_Gradients, out_proj_bias, out_proj_bias_Gradients);
        }
    }



    #region forward and backward propagation

    public override void ForwardPropagation(List<Tensor> allX, Tensor y, bool isTraining)
    {
        Debug.Assert(allX.Count == 3);

        var Q = allX[QUERY_LAYER_INDEX]; // queries: (batch_size, query_timeSteps == input_seq_length, embed_dim)
        var K = allX[KEY_LAYER_INDEX];   // keys:    (batch_size, value_timeSteps, embed_dim)
        var V = allX[VALUE_LAYER_INDEX]; // values:  (batch_size, value_timeSteps, embed_dim)

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
        LinearLayer.DenseForwardPropagation(Q_heads, Q, w_Q, w_Q_bias, flattenInputTensorOnLastDimension);
        GetFloatTensor(ref Q_heads_T , new[] { batch_size, _num_heads, query_time_steps, _key_dim });
        Q_heads.Reshape(batch_size, query_time_steps, _num_heads, _key_dim).TransposeSecondAndThirdDimension(Q_heads_T);
        Q_heads_T.ReshapeInPlace(batch_size*_num_heads, query_time_steps, _key_dim);

        var K_heads = Q_K_V_heads_buffer.Reshape(batch_size * key_value_time_steps, _num_heads * _key_dim);
        LinearLayer.DenseForwardPropagation(K_heads, K, w_K, w_K_bias, flattenInputTensorOnLastDimension);
        GetFloatTensor(ref K_heads_T, new[] { batch_size, _num_heads, key_value_time_steps, _key_dim });
        K_heads.Reshape(batch_size, key_value_time_steps, _num_heads, _key_dim).TransposeSecondAndThirdDimension(K_heads_T);
        K_heads_T.ReshapeInPlace(batch_size * _num_heads, key_value_time_steps, _key_dim);

        var V_heads = Q_K_V_heads_buffer.Reshape(batch_size * key_value_time_steps, _num_heads * _value_dim);
        LinearLayer.DenseForwardPropagation(V_heads, V, w_V, w_V_bias, flattenInputTensorOnLastDimension);
        GetFloatTensor(ref V_heads_T, new[] { batch_size, _num_heads, key_value_time_steps, _value_dim });
        V_heads.Reshape(batch_size, key_value_time_steps, _num_heads, _value_dim).TransposeSecondAndThirdDimension(V_heads_T);
        V_heads_T.ReshapeInPlace(batch_size * _num_heads, key_value_time_steps, _value_dim);

        var attention_heads = Q_K_V_heads_buffer.Reshape(V_heads_T.Shape);
        ScaledDotProductAttentionLayer.ScaledDotProductAttentionForwardPropagation(Q_heads_T, K_heads_T, V_heads_T, attention_heads, isTraining, ref weights_buffer, Network.MemoryPool, use_scale, _is_causal);

        GetFloatTensor(ref attention_heads_T, new[] { batch_size, key_value_time_steps, _num_heads, _value_dim });
        attention_heads.Reshape(batch_size, _num_heads, key_value_time_steps, _value_dim).TransposeSecondAndThirdDimension(attention_heads_T);
        attention_heads_T.ReshapeInPlace(batch_size, key_value_time_steps, _num_heads* _value_dim);
        LinearLayer.DenseForwardPropagation(y, attention_heads_T, out_proj_weight, out_proj_bias, flattenInputTensorOnLastDimension);

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
        ////dy:          (batch_size, value_timeSteps, value_embed_dim)
        var dQ = allDx[QUERY_LAYER_INDEX];    // queries:    (batch_size, query_timeSteps == input_seq_length, embed_dim)
        var dK = allDx[KEY_LAYER_INDEX];       // keys:       (batch_size, value_timeSteps, embed_dim)
        var dV = allDx[VALUE_LAYER_INDEX];     // values:     (batch_size, value_timeSteps, embed_dim)
        //Debug.Assert(weights_buffer != null);
        var batch_size = dK.Shape[0];
        var query_time_steps = dQ.Shape[1];
        var value_time_steps = dV.Shape[1];
        var Q = allX[QUERY_LAYER_INDEX];
        var K = allX[KEY_LAYER_INDEX];
        var V = allX[VALUE_LAYER_INDEX];
        
        var dQ_dK_dV_heads_buffer = GetFloatTensor(new[] { Math.Max(batch_size * query_time_steps * _num_heads * _key_dim, batch_size * value_time_steps * _num_heads * _value_dim) });

        var d_attention_heads_T = dQ_dK_dV_heads_buffer.Reshape(attention_heads_T.Shape);
        LinearLayer.DenseBackwardPropagation(d_attention_heads_T, out_proj_weight_Gradients, out_proj_bias_Gradients,
            attention_heads_T, dy, out_proj_weight, out_proj_bias,
            Network.Sample, 0, false, flattenInputTensorOnLastDimension);

        var d_attention_heads = GetFloatTensor(V_heads_T.Shape);
        d_attention_heads_T.Reshape(batch_size, value_time_steps, _num_heads, _value_dim).TransposeSecondAndThirdDimension(d_attention_heads);
        d_attention_heads.ReshapeInPlace(batch_size * _num_heads, value_time_steps, _value_dim);

        var dQ_heads_T = GetFloatTensor(Q_heads_T.Shape);
        var dK_heads_T = GetFloatTensor(K_heads_T.Shape);
        var dV_heads_T = GetFloatTensor(V_heads_T.Shape);

        ScaledDotProductAttentionLayer.ScaledDotProductAttentionBackwardPropagation( 
            /* Out */ dQ_heads_T, /* Out */ dK_heads_T, /* Out */dV_heads_T, 
            /* In */ Q_heads_T, /* In */ K_heads_T,  /* In */ V_heads_T, /* In */ d_attention_heads, /* In */ ref weights_buffer,
            Network.MemoryPool, use_scale);

        dQ_heads_T.Reshape(batch_size, _num_heads, query_time_steps, _key_dim).TransposeSecondAndThirdDimension(dQ_dK_dV_heads_buffer  /* dQ_heads */);
        LinearLayer.DenseBackwardPropagation(dQ, w_Q_Gradients, w_Q_bias_Gradients, Q, dQ_dK_dV_heads_buffer.Reshape(batch_size * query_time_steps, -1), w_Q, w_Q_bias, Network.Sample, 0, false, flattenInputTensorOnLastDimension);

        dK_heads_T.Reshape(batch_size, _num_heads, query_time_steps, _key_dim).TransposeSecondAndThirdDimension(dQ_dK_dV_heads_buffer  /* dK_heads */);
        LinearLayer.DenseBackwardPropagation(dK, w_K_Gradients, w_K_bias_Gradients, K, dQ_dK_dV_heads_buffer.Reshape(batch_size * query_time_steps, -1), w_K, w_K_bias, Network.Sample, 0, false, flattenInputTensorOnLastDimension);

        dV_heads_T.Reshape(batch_size, _num_heads, value_time_steps, _value_dim).TransposeSecondAndThirdDimension(dQ_dK_dV_heads_buffer /* dV_heads */);
        LinearLayer.DenseBackwardPropagation(dV, w_V_Gradients, w_V_bias_Gradients, V, dQ_dK_dV_heads_buffer.Reshape(batch_size * value_time_steps, -1), w_V, w_V_bias, Network.Sample, 0, false, flattenInputTensorOnLastDimension);

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
    private Optimizer[] AllOptimizer => new[] { _w_Q_optimizer, _w_K_optimizer, _w_V_optimizer, _out_proj_weight_optimizer };


    public override void ResetParameters(bool resetAlsoOptimizerWeights = true)
    {
        //trainable params
        foreach (var w in new[] { w_Q, w_K, w_V, out_proj_weight })
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
            .Add(nameof(_use_bias_Q_K_V), _use_bias_Q_K_V)
            .Add(nameof(_use_bias_O), _use_bias_O)
            .Add(nameof(_is_causal), _is_causal)
            .ToString();
    }
    public static MultiheadAttention Deserialize(IDictionary<string, object> serialized, Network network)
    {
        var num_heads = (int)serialized[nameof(_num_heads)];
        var key_dim = (int)serialized[nameof(_key_dim)];
        var value_dim = (int)serialized[nameof(_value_dim)];
        var use_bias_Q_K_V = (bool)serialized[nameof(_use_bias_Q_K_V)];
        var use_bias_O = (bool)serialized[nameof(_use_bias_O)];
        var is_causal = (bool)serialized[nameof(_is_causal)];
        var previousLayerIndexes = (int[])serialized[nameof(PreviousLayerIndexes)];
        return new MultiheadAttention(num_heads, key_dim, value_dim, use_bias_Q_K_V, use_bias_O, is_causal, previousLayerIndexes[0], previousLayerIndexes[1], previousLayerIndexes[2], network, (string)serialized[nameof(LayerName)]);
    }
    public override void AddToOtherNetwork(Network otherNetwork) { AddToOtherNetwork(otherNetwork, Deserialize); }
    #endregion


    #region Multi GPU Support
    public override void ReplaceParameters(List<Tensor> newParameters)
    {
        FreeFloatTensor(ref _weights);
        _weights = newParameters[0];
        if (_bias != null)
        {
            Debug.Assert(newParameters.Count == 2);
            FreeFloatTensor(ref _bias);
            _bias = newParameters[1];
        }
        else
        {
            Debug.Assert(newParameters.Count == 1);
        }
    }
    public override void ReplaceGradients(List<Tensor> newGradients)
    {
        FreeFloatTensor(ref _weightGradients);
        _weightGradients = newGradients[0];
        if (_biasGradients != null)
        {
            Debug.Assert(newGradients.Count == 2);
            FreeFloatTensor(ref _biasGradients);
            _biasGradients = newGradients[1];
        }
        else
        {
            Debug.Assert(newGradients.Count == 1);
        }
    }
    #endregion

    public override List<Tuple<Tensor, string>> Parameters
    {
        get
        {
            var result = new List<Tuple<Tensor, string>>
            {
                Tuple.Create(in_proj_weight, DatasetNameToDatasetPath("in_proj_weight")),
                Tuple.Create(in_proj_bias, DatasetNameToDatasetPath("in_proj_bias")),
                Tuple.Create(out_proj_weight, DatasetNameToDatasetPath("out_proj.weight")),
                Tuple.Create(out_proj_bias, DatasetNameToDatasetPath("out_proj.bias")),
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
        return PreviousLayers[VALUE_LAYER_INDEX].OutputShape(batchSize);
    }



    #region PyTorch support
    //see : https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
    public override void ToPytorchModule(List<string> constructorLines, List<string> forwardLines)
    {
        var embed_dim = ValueLayer.OutputShape(1)[2];

        if (_is_causal)
        {
            const string attn_mask_dict_line = "self.attn_mask_dict = dict()";
            if (!constructorLines.Contains(attn_mask_dict_line))
            {
                constructorLines.Add(attn_mask_dict_line);
            }
        }

        if (_use_bias_Q_K_V != _use_bias_O)
        {
            throw new ArgumentException($"use_bias_Q_K_V ({_use_bias_Q_K_V}) must be equal to use_bias_O ({_use_bias_O}) in PyTorch");
        }

        constructorLines.Add("self." + LayerName + " = torch.nn.MultiheadAttention("
                             + "embed_dim="+ embed_dim
                             + ", num_heads="+_num_heads
                             + ", dropout=0"
                             + ", bias="+ Utils.ToPython(_use_bias_Q_K_V)
                             + ", add_bias_kv=False"
                             + ", add_zero_attn=False"
                             + ", batch_first=True"
                             + ")");

        forwardLines.Add("attn_mask = None");
        if (_is_causal)
        {
            //we build the attention mask 'attn_mask'
            int L = QueryLayer.OutputShape(1)[1]; // Target sequence Length
            int S = ValueLayer.OutputShape(1)[1]; // Source sequence Length
            if (S != L)
            {
                throw new ArgumentException($"Source sequence Length ({S}) must be equal to Target sequence Length ({L}) to build attention mask");
            }
            forwardLines.Add("");
            forwardLines.Add("# We build the attention mask 'attn_mask'");
            forwardLines.Add("if not isinstance(y_conv1D_Q, torch.fx.proxy.Proxy):");
            forwardLines.Add("    sz = y_conv1D_Q.size(1)  # L: Target sequence Length");
            forwardLines.Add("    if sz not in self.attn_mask_dict: self.attn_mask_dict[sz] = torch.nn.Transformer.generate_square_subsequent_mask(sz, device = x.device, dtype = x.dtype)");
            forwardLines.Add("    attn_mask = self.attn_mask_dict[sz]");
        }
        forwardLines.Add(GetPyTorchOutputVariableName() + ", _ = self." + LayerName+"("
                                                        + QueryLayer.GetPyTorchOutputVariableName()
                                                        + ", " + KeyLayer.GetPyTorchOutputVariableName()
                                                        + ", " + ValueLayer.GetPyTorchOutputVariableName()
                                                        + ", key_padding_mask=None"
                                                        + ", need_weights=False"
                                                        + ", attn_mask=attn_mask"
                                                        + ", average_attn_weights=False"
                                                        + ", is_causal=" + Utils.ToPython(_is_causal)
                                                        + ")");
    }

    #endregion


    public Tensor w_Q => _weights.Slice(0, _shapes_w_Q_K_V_O[0]); // (num_heads*key_dim, embed_dim)
    public Tensor w_K => _weights.Slice(_count_w_Q_K_V_O[0], _shapes_w_Q_K_V_O[1]); // (num_heads*key_dim, embed_dim)
    public Tensor w_V => _weights.Slice(_count_w_Q_K_V_O[0] + _count_w_Q_K_V_O[1], _shapes_w_Q_K_V_O[2]); // (num_heads*value_dim, embed_dim)
    public Tensor in_proj_weight => _weights.Slice(0, new []{ _shapes_w_Q_K_V_O[0][0] + _shapes_w_Q_K_V_O[1][0]+ _shapes_w_Q_K_V_O[2][0], _shapes_w_Q_K_V_O[0][1]});
    public Tensor out_proj_weight => _weights.Slice(_count_w_Q_K_V_O[0] + _count_w_Q_K_V_O[1] + _count_w_Q_K_V_O[2], _shapes_w_Q_K_V_O[3]); // (num_heads*value_dim, embed_dim)
    private Tensor w_Q_bias => !_use_bias_Q_K_V ? null:_bias?.Slice(0, _shapes_w_bias_Q_K_V_O[0]); // (1, num_heads*key_dim)
    private Tensor w_K_bias => !_use_bias_Q_K_V ? null : _bias?.Slice(_count_w_bias_Q_K_V_O[0], _shapes_w_bias_Q_K_V_O[1]); // (1, num_heads*key_dim)
    private Tensor w_V_bias => !_use_bias_Q_K_V ? null : _bias?.Slice(_count_w_bias_Q_K_V_O[0] + _count_w_bias_Q_K_V_O[1], _shapes_w_bias_Q_K_V_O[2]); // (1, num_heads*value_dim)
    private Tensor in_proj_bias => !_use_bias_Q_K_V ? null : _bias?.Slice(0, new[] { _count_w_bias_Q_K_V_O[0] + _count_w_bias_Q_K_V_O[1] + _count_w_bias_Q_K_V_O[2] });
    private Tensor out_proj_bias => !_use_bias_O ?null: _bias?.Slice(_count_w_bias_Q_K_V_O[0] + _count_w_bias_Q_K_V_O[1] + _count_w_bias_Q_K_V_O[2], _shapes_w_bias_Q_K_V_O[3]); // (1, embed_dim)
    private Tensor w_Q_Gradients => _weightGradients.Slice(0, _shapes_w_Q_K_V_O[0]); // (num_heads*key_dim, embed_dim)
    private Tensor w_K_Gradients => _weightGradients.Slice(_count_w_Q_K_V_O[0], _shapes_w_Q_K_V_O[1]); // (num_heads*key_dim, embed_dim)
    private Tensor w_V_Gradients => _weightGradients.Slice(_count_w_Q_K_V_O[0] + _count_w_Q_K_V_O[1], _shapes_w_Q_K_V_O[2]); // (num_heads*value_dim, embed_dim)
    private Tensor out_proj_weight_Gradients => _weightGradients.Slice(_count_w_Q_K_V_O[0] + _count_w_Q_K_V_O[1] + _count_w_Q_K_V_O[2], _shapes_w_Q_K_V_O[3]); // (embed_dim, num_heads*value_dim )
    private Tensor w_Q_bias_Gradients => !_use_bias_Q_K_V ? null : _biasGradients?.Slice(0, _shapes_w_bias_Q_K_V_O[0]); // (1, num_heads*key_dim)
    private Tensor w_K_bias_Gradients => !_use_bias_Q_K_V ? null : _biasGradients?.Slice(_count_w_bias_Q_K_V_O[0], _shapes_w_bias_Q_K_V_O[1]); // (1, num_heads*key_dim)
    private Tensor w_V_bias_Gradients => !_use_bias_Q_K_V ? null : _biasGradients?.Slice(_count_w_bias_Q_K_V_O[0] + _count_w_bias_Q_K_V_O[1], _shapes_w_bias_Q_K_V_O[2]); // (1, num_heads*value_dim)
    private Tensor out_proj_bias_Gradients => !_use_bias_O?null: _biasGradients?.Slice(_count_w_bias_Q_K_V_O[0] + _count_w_bias_Q_K_V_O[1] + _count_w_bias_Q_K_V_O[2], _shapes_w_bias_Q_K_V_O[3]); // (1, embed_dim)

    private Layer QueryLayer => PreviousLayers[QUERY_LAYER_INDEX];
    private Layer KeyLayer => PreviousLayers[KEY_LAYER_INDEX];
    private Layer ValueLayer => PreviousLayers[VALUE_LAYER_INDEX];

}
