using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using SharpNet.Data;
using SharpNet.GPU;
using SharpNet.Networks;

namespace SharpNet.Layers;


/// <summary>
/// Input:
///  allX[0] = Q of shape (batch_size, query_timeSteps == input_seq_length, embedding_dim)
///  allX[1] = K of shape (batch_size, value_timeSteps, embedding_dim)
///  allX[2] = V of shape (batch_size, value_timeSteps, embedding_dim)
///     in most cases, K and V are the same tensor (K is optional, it's default value is V)
/// Output:
/// y of shape:           (batch_size, value_timeSteps, embedding_dim)
///                       (same as V shape)
/// </summary>
public class ScaledDotProductAttentionLayer : Layer
{
    private readonly bool _use_scale;
    private readonly bool _is_causal;

    private Tensor _weights_buffer = null;
    public ScaledDotProductAttentionLayer(bool use_scale, bool is_causal, int queriesLayerIndex, int keysLayerIndex, int valuesLayerIndex,
        Network network, string layerName = "") : base(network, new[]{queriesLayerIndex, keysLayerIndex, valuesLayerIndex }, layerName)
    {
        _use_scale = use_scale;
        _is_causal = is_causal;
    }

    #region forward and backward propagation
    public override void ForwardPropagation(List<Tensor> allX, Tensor y, bool isTraining)
    {
        Debug.Assert(allX.Count == 3);
        var Q = allX[QUERY_LAYER_INDEX];      // queries: (batch_size, query_timeSteps == input_seq_length, embedding_dim)
        var K = allX[KEY_LAYER_INDEX];        // keys:    (batch_size, value_timeSteps, embedding_dim)
        var V = allX[VALUE_LAYER_INDEX];      // values:  (batch_size, value_timeSteps, embedding_dim)

        ScaledDotProductAttentionForwardPropagation(Q, K, V, y, isTraining, ref _weights_buffer, Network.MemoryPool, _use_scale, _is_causal);
    }

    public static void ScaledDotProductAttentionForwardPropagation(Tensor Q, Tensor K, Tensor V, Tensor y, bool isTraining, ref Tensor _weights_buffer, TensorMemoryPool memoryPool, bool use_scale, bool is_causal)
    {

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
        var value_time_steps = V.Shape[1];
        var embedding_dim = K.Shape[2];

        //Scoring the queries against the keys after transposing the latter, and scaling
        //scores = matmul(Q, K, transpose_keys = True) / math.sqrt(embedding_dim))
        var scores_buffer = memoryPool.GetFloatTensor(new[] { batch_size, query_time_steps, value_time_steps });
        float scaling = (use_scale) ? (1.0f / MathF.Sqrt(embedding_dim)) : 1.0f;

        scores_buffer.BatchMatrixMultiplication(Q, false, K, true, scaling, 0.0f);

        if (is_causal)
        {
            scores_buffer.SetAllElementsAboveMainDiagonal(-1e12f);
        }

        //Computing the weights by a softmax operation
        //weights = softmax(scores)
        memoryPool.GetFloatTensor(ref _weights_buffer, scores_buffer.Shape);       // (batch_size, query_time_steps, value_time_steps)
        scores_buffer.ActivationForward(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX_LAST_DIMENSION, null, _weights_buffer);

        //Computing the attention by a weighted sum of the value vectors
        //y = matmul(weights, V)
        y.BatchMatrixMultiplication(_weights_buffer, false, V, false, 1.0f, 0.0f);
        memoryPool.FreeFloatTensor(scores_buffer);
        if (!isTraining)
        {
            memoryPool.FreeFloatTensor(ref _weights_buffer);
        }
    }

    private const int QUERY_LAYER_INDEX = 0;
    private const int KEY_LAYER_INDEX = 1;
    private const int VALUE_LAYER_INDEX = 2;


    public override void BackwardPropagation(List<Tensor> allX, Tensor y_NotUsed, Tensor dy, List<Tensor> allDx)
    {
        //dy:          (batch_size, value_timeSteps, value_embedding_dim)
        var dQ = allDx[QUERY_LAYER_INDEX];     // queries:    (batch_size, query_timeSteps == input_seq_length, embedding_dim)
        var dK = allDx[KEY_LAYER_INDEX];       // keys:       (batch_size, value_timeSteps, embedding_dim)
        var dV = allDx[VALUE_LAYER_INDEX];     // values:     (batch_size, value_timeSteps, embedding_dim)
        var Q = allX[QUERY_LAYER_INDEX];
        var K = allX[KEY_LAYER_INDEX];
        var V = allX[VALUE_LAYER_INDEX];
        ScaledDotProductAttentionBackwardPropagation(dQ, dK, dV, Q, K, V, dy, ref _weights_buffer, Network.MemoryPool, _use_scale);
    }

    
    public static void ScaledDotProductAttentionBackwardPropagation(/* Out */ Tensor dQ, /* Out */ Tensor dK, /* Out */ Tensor dV, /* In */ Tensor Q, /* In */ Tensor K, /* In */ Tensor V, /* In */ Tensor dy, /* In */ ref Tensor weights_buffer, TensorMemoryPool memoryPool, bool use_scale)
    {
        Debug.Assert(weights_buffer != null);
        //dy:          (batch_size, value_timeSteps, value_embedding_dim)
        var embedding_dim = dV.Shape[2];

        dV.BatchMatrixMultiplication(weights_buffer, true, dy, false, 1.0f, 0.0f);

        float scaling = (use_scale) ? (1.0f / MathF.Sqrt(embedding_dim)) : 1.0f;

        var scores_gradients_buffer = memoryPool.GetFloatTensor(weights_buffer.Shape);       // (batch_size, query_time_steps, value_time_steps)
        //1st step: we store in 'scores_gradients_buffer' the weights_gradients_buffer
        scores_gradients_buffer.BatchMatrixMultiplication(dy, false, V, true, scaling, 0.0f);
        //2nd step: we store in 'scores_gradients_buffer' the proba distribution (scores)
        scores_gradients_buffer.ActivationBackward(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX_LAST_DIMENSION, null, scores_gradients_buffer /* dy*/, null /*x*/, weights_buffer /*y*/);

        dQ.BatchMatrixMultiplication(scores_gradients_buffer, false, K, false, 1, 0.0f);
        dK.BatchMatrixMultiplication(scores_gradients_buffer, true, Q, false, 1, 0.0f);

        memoryPool.FreeFloatTensor(scores_gradients_buffer);
        memoryPool.FreeFloatTensor(ref weights_buffer);
    }

  
    private Layer QueryLayer => PreviousLayers[QUERY_LAYER_INDEX];
    private Layer KeyLayer => PreviousLayers[KEY_LAYER_INDEX];
    private Layer ValueLayer => PreviousLayers[VALUE_LAYER_INDEX];
    public override bool OutputNeededForBackwardPropagation => false;
    public override bool InputNeededForBackwardPropagation => true;
    #endregion

    #region serialization

    public override string Serialize()
    {
        return RootSerializer()
            .Add(nameof(_use_scale), _use_scale)
            .Add(nameof(_is_causal), _is_causal)
            .ToString();
    }
    public static ScaledDotProductAttentionLayer Deserialize(IDictionary<string, object> serialized, Network network)
    {
        var useScale = (bool)serialized[nameof(_use_scale)];
        var useCausalMask = (bool)serialized[nameof(_is_causal)];
        var previousLayerIndexes = (int[])serialized[nameof(PreviousLayerIndexes)];
        return new ScaledDotProductAttentionLayer(useScale, useCausalMask, previousLayerIndexes[0], previousLayerIndexes[1], previousLayerIndexes[2], network, (string)serialized[nameof(LayerName)]);
    }
    public override void AddToOtherNetwork(Network otherNetwork) { AddToOtherNetwork(otherNetwork, Deserialize); }
    #endregion


    #region PyTorch support
    //see : https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    public override void ToPytorchModule(List<string> constructorLines, List<string> forwardLines)
    {
        forwardLines.Add(GetPyTorchOutputVariableName() + " = F.scaled_dot_product_attention("
                                                        + QueryLayer.GetPyTorchOutputVariableName()
                                                        + ", " + KeyLayer.GetPyTorchOutputVariableName()
                                                        + ", " + ValueLayer.GetPyTorchOutputVariableName()
                                                        + ", attn_mask=None"
                                                        + ", dropout_p=0.0"
                                                        + ", is_causal="+ Utils.ToPython(_is_causal)
                                                        + ", scale="+ (_use_scale?"None":"1")
                                                        + ")");
    }

    #endregion

    protected override List<Tensor> EmbeddedTensors(bool includeOptimizeTensors)
    {
        var result = base.EmbeddedTensors(includeOptimizeTensors);
        result.AddRange(new[] { _weights_buffer});
        result.RemoveAll(t => t == null);
        return result;
    }
    
    public override int[] OutputShape(int batchSize)
    {
        return ValueLayer.OutputShape(batchSize);
    }
}
