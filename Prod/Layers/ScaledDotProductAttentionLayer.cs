﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using SharpNet.Data;
using SharpNet.GPU;
using SharpNet.Networks;

namespace SharpNet.Layers;

public class ScaledDotProductAttentionLayer : Layer
{
    private readonly bool _useScale;

    private Tensor _weights_buffer = null;
    public ScaledDotProductAttentionLayer(bool use_scale, int queriesLayerIndex, int valuesLayerIndex, int keysLayerIndex,
        Network network, string layerName = "") : base(network, new[]{queriesLayerIndex, valuesLayerIndex, keysLayerIndex }, layerName)
    {
        _useScale = use_scale;
    }

    #region forward and backward propagation
    public override void ForwardPropagation(List<Tensor> allX, Tensor y, bool isTraining)
    {
        Debug.Assert(allX.Count == 3);

        var Q = allX[QUERIES_LAYER_INDEX];    // queries: (batch_size, timeSteps == input_seq_length, query_embedding_dim)
        var V = allX[VALUES_LAYER_INDEX];    // values:  (batch_size, timeSteps, value_embedding_dim)
        var K = allX[KEYS_LAYER_INDEX];    // keys:    (batch_size, timeSteps, key_embedding_dim)

        if (!Q.Shape.SequenceEqual(K.Shape))
        {
            throw new ArgumentException($"Q.Shape and K.Shape must be equal, but are {Tensor.ShapeToString(Q.Shape)} and {Tensor.ShapeToString(K.Shape)}");
        }
        if (!V.Shape.SequenceEqual(y.Shape))
        {
            throw new ArgumentException($"V.Shape and y.Shape must be equal, but are {Tensor.ShapeToString(V.Shape)} and {Tensor.ShapeToString(y.Shape)}");
        }
        if (Q.Shape[1] != V.Shape[1])
        {
            throw new ArgumentException($"queries.Shape[1] and values.Shape[1] must be equal (same timeSteps), but are {Q.Shape[1]} and {V.Shape[1]}");
        }
        var batch_size = K.Shape[0];
        var timesteps = K.Shape[1];
        var keys_embedding_dim = K.Shape[2];

        //Scoring the queries against the keys after transposing the latter, and scaling
        //scores = matmul(Q, K, transpose_keys = True) / math.sqrt(key_embedding_dim))
        var scores_buffer  = GetFloatTensor(new[] { batch_size, timesteps, timesteps });        // (batch_size, timeSteps, timeSteps)
        float scaling = (_useScale)?(1.0f / MathF.Sqrt(keys_embedding_dim)) :1.0f;

        scores_buffer.BatchMatrixMultiplication(Q, false, K, true, scaling, 0.0f);

        //TODO : use masking

        //Computing the weights by a softmax operation
        //weights = softmax(scores)
        GetFloatTensor(ref _weights_buffer, scores_buffer.Shape);       // (batch_size, timeSteps, timeSteps)
        var weights2DShape = new [] { batch_size * timesteps, timesteps };
        scores_buffer.WithNewShape(weights2DShape).ActivationForward(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX, null, _weights_buffer.WithNewShape(weights2DShape));

        //Computing the attention by a weighted sum of the value vectors
        //y = matmul(weights, V)
        y.BatchMatrixMultiplication(_weights_buffer, false, V, false, 1.0f, 0.0f);

        FreeFloatTensor(scores_buffer);
        if (!isTraining)
        {
            FreeFloatTensor(ref _weights_buffer);
        }
    }


    private const int QUERIES_LAYER_INDEX = 0;
    private const int VALUES_LAYER_INDEX = 1;
    private const int KEYS_LAYER_INDEX = 2;

    public override void BackwardPropagation(List<Tensor> allX, Tensor y_NotUsed, Tensor dy, List<Tensor> allDx)
    {
        //dy:          (batch_size, timeSteps, value_embedding_dim)
        var dQ = allDx[QUERIES_LAYER_INDEX];    // queries:    (batch_size, timeSteps == input_seq_length, key_embedding_dim)
        var dV = allDx[VALUES_LAYER_INDEX];     // values:     (batch_size, timeSteps, value_embedding_dim)
        var dK = allDx[KEYS_LAYER_INDEX];       // keys:       (batch_size, timeSteps, key_embedding_dim)
        Debug.Assert(_weights_buffer != null);
        var batch_size = dK.Shape[0];
        var time_steps = dK.Shape[1];
        var Q = allX[QUERIES_LAYER_INDEX];
        var V = allX[VALUES_LAYER_INDEX];
        var K = allX[KEYS_LAYER_INDEX];

        dV.BatchMatrixMultiplication(_weights_buffer, true, dy, false, 1.0f, 0.0f);

        var weights_gradients_buffer = GetFloatTensor(_weights_buffer.Shape);       // (batch_size, timeSteps, timeSteps)
        weights_gradients_buffer.BatchMatrixMultiplication(dy, false, V, true, 1.0f, 0.0f);
        var scores_gradients_buffer = GetFloatTensor(_weights_buffer.Shape);       // (batch_size, timeSteps, timeSteps)
        var weights2DShape = new[] { batch_size * time_steps, time_steps };
        scores_gradients_buffer.WithNewShape(weights2DShape).ActivationBackward(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX, null, weights_gradients_buffer.WithNewShape(weights2DShape) /* dy*/, null /*x*/, _weights_buffer.WithNewShape(weights2DShape) /*y*/);
        dQ.BatchMatrixMultiplication(scores_gradients_buffer, false, K, false, 1.0f, 0.0f);
        dK.BatchMatrixMultiplication(scores_gradients_buffer, true, Q, false, 1.0f, 0.0f);

        FreeFloatTensor(ref weights_gradients_buffer);
        FreeFloatTensor(ref scores_gradients_buffer);

        FreeFloatTensor(ref _weights_buffer);
    }
    private Layer ValueLayer => PreviousLayers[VALUES_LAYER_INDEX];
    public override bool OutputNeededForBackwardPropagation => false;
    public override bool InputNeededForBackwardPropagation => true;
    #endregion

    #region serialization

    public override string Serialize()
    {
        return RootSerializer()
            .Add(nameof(_useScale), _useScale)
            .ToString();
    }
    public static ScaledDotProductAttentionLayer Deserialize(IDictionary<string, object> serialized, Network network)
    {
        var useScale = (bool)serialized[nameof(_useScale)];
        var previousLayerIndexes = (int[])serialized[nameof(PreviousLayerIndexes)];
        return new ScaledDotProductAttentionLayer(useScale, previousLayerIndexes[0], previousLayerIndexes[1], previousLayerIndexes[2], network, (string)serialized[nameof(LayerName)]);
    }
    public override void AddToOtherNetwork(Network otherNetwork) { AddToOtherNetwork(otherNetwork, Deserialize); }
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
