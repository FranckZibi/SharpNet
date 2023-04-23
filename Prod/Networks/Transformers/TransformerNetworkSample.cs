using System;
using SharpNet.Datasets;
using SharpNet.GPU;
using SharpNet.Layers;

// ReSharper disable FieldCanBeMadeReadOnly.Global
// ReSharper disable MemberCanBePrivate.Global

namespace SharpNet.Networks.Transformers;

public class TransformerNetworkSample : NetworkSample
{
    // ReSharper disable once EmptyConstructor
    public TransformerNetworkSample()
    {
    }

    #region Hyper-Parameters

    public float layer_norm_epsilon = LayerNormalizationLayer.DEFAULT_EPSILON;
    public int embedding_dim = -1; // == d_model
    public cudnnActivationMode_t lastActivationLayer = cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX_LAST_DIMENSION;
    public int N_PositionalEncoding = PositionalEncodingAttnIsAllYouNeedLayer.DEFAULT_N_POSITIONAL_ENCODING;
    public bool layer_norm_before_last_dense = true; // must be true

    //encoders Hyper-Parameters
    public int encoder_num_transformer_blocks = -1;
    public int encoder_num_heads = -1; //must be a divider of 'embedding_dim'
    public bool encoder_mha_use_bias_Q_V_K = true;
    public bool encoder_mha_use_bias_O = true;
    public float encoder_mha_dropout = 0.0f;
    public int encoder_feed_forward_dim = -1;
    public float encoder_feed_forward_dropout = 0.0f;
    public bool encoder_use_causal_mask = false;
    public bool encoder_add_layer_norm_before_mha = true;   //should be true
    public bool encoder_add_layer_norm_after_mha = false;   //should be false

    public bool layer_norm_before_ffd  = true;              //should be true
    public bool layer_norm_after_ffd  = false;              //should be false


    //decoders Hyper-Parameters
    public int decoder_num_transformer_blocks = -1;
    public int decoder_num_heads = -1; //must be a divider of 'embedding_dim'
    public bool decoder_mha_use_bias_Q_V_K = true;
    public bool decoder_mha_use_bias_O = true;
    public float decoder_mha_dropout = 0.0f;
    public int decoder_feed_forward_dim = -1;
    public float decoder_feed_forward_dropout = 0.0f;
    public bool decoder_add_layer_norm_before_mha = true;
    public bool decoder_add_layer_norm_after_mha = false;

    #endregion

    //public static TransformerNetworkSample DefaultFullEncoders(int embedding_dim, int N, int num_heads, bool use_bias, float mha_dropout, int feed_forward_dim, float feed_forward_dropout)
    //{
    //    var sample = (TransformerNetworkSample)new TransformerNetworkSample()
    //        {
    //            LossFunction = EvaluationMetricEnum.CategoricalCrossentropy,
    //            CompatibilityMode = CompatibilityModeEnum.TensorFlow,
    //            lambdaL2Regularization = 0.0005,
    //            NumEpochs = 150,
    //            BatchSize = 128,
    //            InitialLearningRate = 0.1,

    //            //No Data augmentation
    //            DataAugmentationType = ImageDataGenerator.DataAugmentationEnum.NO_AUGMENTATION,
              
    //        }
    //        .WithCommonParams(embedding_dim)
    //        .WithEncoders(N, num_heads, use_bias, mha_dropout, feed_forward_dim, feed_forward_dropout)
    //        .WithSGD(0.9, false)
    //        .WithCyclicCosineAnnealingLearningRateScheduler(10, 2);
    //    return sample;
    //}

    public override void BuildLayers(Network nn, AbstractDatasetSample datasetSample)
    {
        //nn.PropagationManager.LogPropagation = true;
        //nn.Sample.DisplayTensorContentStats = true;

        if (encoder_num_transformer_blocks >= 1 && decoder_num_transformer_blocks<=0)
        {
            //Full encoders
            var inputShape = datasetSample.GetInputShapeOfSingleElement();
            if (inputShape.Length != 1)
            {
                throw new ArgumentException($"inputShape.Length={inputShape.Length} != 1");
            }
            int timeSteps = inputShape[0];
            nn.Input(timeSteps, -1, -1)
                .Embedding(new[] { datasetSample.NumClass }, new[] { embedding_dim }, new[] { -1 }, 0.0)
                .PositionalEncodingAttnIsAllYouNeedLayer(N_PositionalEncoding)
                ;
            AddTransformers(nn, datasetSample.NumClass, nn.LastLayerIndex, -1);
            return;
        }

        throw new NotSupportedException("only full encoders are currently supported");
    }

    //private TransformerNetworkSample WithCommonParams(int embedding_dim, cudnnActivationMode_t lastActivationLayer = cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX, float layer_norm_epsilon = 1e-6f, int N_PositionalEncoding = PositionalEncodingAttnIsAllYouNeedLayer.DEFAULT_N_POSITIONAL_ENCODING)
    //{
    //    this.embedding_dim = embedding_dim;
    //    this.lastActivationLayer = lastActivationLayer;
    //    this.layer_norm_epsilon = layer_norm_epsilon;
    //    this.N_PositionalEncoding = N_PositionalEncoding;
    //    return this;
    //}

    //private TransformerNetworkSample WithEncoders(int N, int num_heads, bool use_bias, float mha_dropout, int feed_forward_dim, float feed_forward_dropout, bool use_causal_mask = false, bool add_layer_norm_before_mha = false, bool add_layer_norm_after_mha = true)
    //{
    //    encoder_num_transformer_blocks = N;
    //    encoder_num_heads = num_heads;
    //    encoder_mha_use_bias = use_bias;
    //    encoder_mha_dropout = mha_dropout;
    //    encoder_feed_forward_dim = feed_forward_dim;
    //    encoder_feed_forward_dropout = feed_forward_dropout;
    //    encoder_use_causal_mask = use_causal_mask;
    //    encoder_add_layer_norm_after_mha = add_layer_norm_after_mha;
    //    return this;
    //}
    //private TransformerNetworkSample WithDecoders(int N, int num_heads, bool use_bias, float mha_dropout, int feed_forward_dim, float feed_forward_dropout, bool add_layer_norm_before_mha, bool add_layer_norm_after_mha)
    //{
    //    decoder_num_transformer_blocks = N;
    //    decoder_num_heads = num_heads;
    //    decoder_mha_use_bias = use_bias;
    //    decoder_mha_dropout = mha_dropout;
    //    decoder_feed_forward_dim = feed_forward_dim;
    //    decoder_feed_forward_dropout = feed_forward_dropout;
    //    decoder_add_layer_norm_before_mha = add_layer_norm_before_mha;
    //    decoder_add_layer_norm_after_mha = add_layer_norm_after_mha;
    //    return this;
    //}
    //private TransformerNetworkSample DisableEncoders()
    //{
    //    encoder_num_transformer_blocks = encoder_num_heads = encoder_feed_forward_dim = -1;
    //    encoder_mha_dropout = encoder_feed_forward_dropout = 0.0f;
    //    encoder_mha_use_bias = true;
    //    return this;
    //}
    //private TransformerNetworkSample DisableDecoders()
    //{
    //    decoder_num_transformer_blocks = decoder_num_heads = decoder_feed_forward_dim = -1;
    //    decoder_mha_dropout = decoder_feed_forward_dropout = 0.0f;
    //    decoder_mha_use_bias = true;
    //    decoder_add_layer_norm_before_mha = false;
    //    decoder_add_layer_norm_after_mha = true;
    //    return this;
    //}
    private void AddTransformers(Network network, int categoryCount, int layerIndexInputEmbedding, int layerIndexOutputEmbedding)
    {
        AddEncoders(network, layerIndexInputEmbedding);
        AddDecoders(network, layerIndexOutputEmbedding, network.LastLayerIndex);
        if (layer_norm_before_last_dense)
        {
            network.LayerNorm(1, layer_norm_epsilon, "last_layer_norm");
        }
        network.Dense(categoryCount, network.Sample.lambdaL2Regularization, true, "probs");
        network.Activation(lastActivationLayer);
    }


    private void AddEncoders(Network network, int layerIndexInputEmbedding)
    {
        if (encoder_num_transformer_blocks <= 0)
        {
            return;
        }
        AssertValidParameters();
        var shapeInputEmbeddingLayer = network.Layers[layerIndexInputEmbedding].OutputShape(1);
        if (shapeInputEmbeddingLayer.Length != 3)
        {
            throw new ArgumentException($"The Input Embedding Layer (before encoder) must have a shape of 3 dimensions (batch, sequence, embedding) but has a shape of {shapeInputEmbeddingLayer.Length} dimensions");
        }
        if (shapeInputEmbeddingLayer[2] != embedding_dim)
        {
            throw new ArgumentException($"The observed embedding_dim of the Input Embedding Layer {shapeInputEmbeddingLayer[2]} is not equal to the expected one ({nameof(embedding_dim)}={embedding_dim}");
        }
        for (int encoderBlock = 0; encoderBlock < encoder_num_transformer_blocks; ++encoderBlock)
        {
            var inputLayerIndex = (encoderBlock ==0) ? layerIndexInputEmbedding : network.LastLayerIndex;
            var layerPrefix = "encoder_" + encoderBlock + "_";

            // Multi-Head Self Attention in Encoder
            AddMultiHeadAttentionBlock(network, layerPrefix+"mha_",
                network.LastLayerIndex, encoder_num_heads, 
                encoder_mha_use_bias_Q_V_K, encoder_mha_use_bias_O, encoder_use_causal_mask, inputLayerIndex, inputLayerIndex, inputLayerIndex, encoder_mha_dropout, layer_norm_epsilon, encoder_add_layer_norm_before_mha, encoder_add_layer_norm_after_mha);

            //Feed Forward in Encoder
            AddFeedForwardBlock(network, layerPrefix+"ffd_", embedding_dim, encoder_feed_forward_dim, encoder_feed_forward_dropout, layer_norm_epsilon, layer_norm_before_ffd, layer_norm_after_ffd);
        }
    }

    private void AddDecoders(Network network, int layerIndexOutputEmbedding, int layerIndexOutputEncoders)
    {
        if (decoder_num_transformer_blocks <= 0)
        {
            return;
        }
        AssertValidParameters();
        var shapeEncodersOutput = network.Layers[layerIndexOutputEncoders].OutputShape(1);
        if (shapeEncodersOutput.Length != 3)
        {
            throw new ArgumentException($"The encoders output must have a shape of 3 dimensions (batch, sequence, embedding) but has a shape of {shapeEncodersOutput.Length} dimensions");
        }
        if (shapeEncodersOutput[2] != embedding_dim)
        {
            throw new ArgumentException($"The observed embedding_dim of the encoders output {shapeEncodersOutput[2]} is not equal to the expected one ({nameof(embedding_dim)}={embedding_dim}");
        }
        var shapeOutputEmbeddingLayer = network.Layers[layerIndexOutputEncoders].OutputShape(1);
        if (shapeOutputEmbeddingLayer.Length != 3)
        {
            throw new ArgumentException($"The Output Embedding Layer (before encoder) must have a shape of 3 dimensions (batch, sequence, embedding) but has a shape of {shapeOutputEmbeddingLayer.Length} dimensions");
        }
        if (shapeOutputEmbeddingLayer[2] != embedding_dim)
        {
            throw new ArgumentException($"The observed embedding_dim of the Output Embedding Layer {shapeOutputEmbeddingLayer[2]} is not equal to the expected one ({nameof(embedding_dim)}={embedding_dim}");
        }
        for (int decoderBlock = 0; decoderBlock < decoder_num_transformer_blocks; ++decoderBlock)
        {
            var layerPrefix = "decoder_" + decoderBlock + "_";

            // Masked Multi-Head Self Attention in Decoder
            AddMultiHeadAttentionBlock(network, layerPrefix+"_masked_mha",
                layerIndexOutputEmbedding, decoder_num_heads, 
                decoder_mha_use_bias_Q_V_K, decoder_mha_use_bias_O,true, layerIndexOutputEmbedding, layerIndexOutputEmbedding, layerIndexOutputEmbedding, decoder_mha_dropout, layer_norm_epsilon, encoder_add_layer_norm_before_mha, encoder_add_layer_norm_after_mha);

            // Multi-Head Cross Attention in Decoder
            AddMultiHeadAttentionBlock(network, layerPrefix+"_mha",
                network.LastLayerIndex, decoder_num_heads, 
                decoder_mha_use_bias_Q_V_K, decoder_mha_use_bias_O, false, network.LastLayerIndex, layerIndexOutputEncoders, layerIndexOutputEncoders, decoder_mha_dropout, layer_norm_epsilon, decoder_add_layer_norm_before_mha, decoder_add_layer_norm_after_mha);

            //Feed Forward in Decoder
            AddFeedForwardBlock(network, layerPrefix+"_ffd", embedding_dim, decoder_feed_forward_dim, decoder_feed_forward_dropout, layer_norm_epsilon, layer_norm_before_ffd, layer_norm_after_ffd);
        }
    }

    private static void AddMultiHeadAttentionBlock(Network network, string layerPrefix, int inputLayerIndex, int num_heads,
        bool use_bias_Q_V_K, bool use_bias_O, bool use_causal_mask,
        int queriesLayerIndex, int valuesLayerIndex, int keysLayerIndex, float dropoutRate, float layer_norm_epsilon, bool addLayerNormBeforeMha, bool addLayerNormAfterMha)
    {
        if (addLayerNormBeforeMha)
        {
            if (queriesLayerIndex != valuesLayerIndex || queriesLayerIndex != keysLayerIndex)
            {
                throw new ArgumentException($"The queries, values and keys must be the same layer when using layer normalization before the multi-head attention");
            }
            network.LayerNorm(1, layer_norm_epsilon, layerPrefix + "layer_norm_before", valuesLayerIndex);
            queriesLayerIndex = valuesLayerIndex = keysLayerIndex = network.LastLayerIndex;
        }
        int embedding_dim = network.Layers[valuesLayerIndex].OutputShape(1)[2];
        int key_dim = embedding_dim / num_heads;
        int value_dim = embedding_dim / num_heads;
        network.MultiHeadAttention(num_heads, key_dim, value_dim, use_bias_Q_V_K, use_bias_O, use_causal_mask, queriesLayerIndex, valuesLayerIndex, keysLayerIndex, layerPrefix + "mha");
        network.Dropout(dropoutRate, layerPrefix + "dropout");
        network.AddLayer(network.LastLayerIndex, inputLayerIndex, layerPrefix + "add");
        if (addLayerNormAfterMha)
        {
            network.LayerNorm(1, layer_norm_epsilon, layerPrefix + "layer_norm_after");
        }
    }
    private static void AddFeedForwardBlock(Network network, string layerPrefix, int embedding_dim, int feed_forward_dim, float feed_forward_dropoutRate, float layer_norm_epsilon, bool addLayerNormBeforeFeedForward, bool addLayerNormAfterFeedForward)
    {
        var inputLayerIndex = network.LastLayerIndex;
        if (addLayerNormBeforeFeedForward)
        {
            network.LayerNorm(1, layer_norm_epsilon, layerPrefix + "layer_norm_before");
        }

        network.Dense(feed_forward_dim, network.Sample.lambdaL2Regularization, true, layerPrefix + "dense1");
        network.Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU, layerPrefix + "activation");
        network.Dense(embedding_dim, network.Sample.lambdaL2Regularization, true, layerPrefix + "dense2");
        
        network.Dropout(feed_forward_dropoutRate, layerPrefix + "dropout");
        network.AddLayer(network.LastLayerIndex, inputLayerIndex, layerPrefix + "add");
        if (addLayerNormAfterFeedForward)
        {
            network.LayerNorm(1, layer_norm_epsilon, layerPrefix + "layer_norm_after");
        }
    }
    private void AssertValidParameters()
    {
        if (embedding_dim <= 0)
        {
            throw new ArgumentException($"{nameof(embedding_dim)} must be >0");
        }
        
        if (encoder_num_transformer_blocks >= 1)
        {
            if (encoder_num_heads <= 0)
            {
                throw new ArgumentException($"{nameof(encoder_num_heads)} must be >0");
            }
            if (encoder_feed_forward_dim <= 0)
            {
                throw new ArgumentException($"{nameof(encoder_feed_forward_dim)} must be >0");
            }
            if (embedding_dim % encoder_num_heads != 0)
            {
                throw new ArgumentException($"{nameof(embedding_dim)} ({embedding_dim}) must be a multiple of {nameof(encoder_num_heads)} ({encoder_num_heads})");
            }
        }
        
        if (decoder_num_transformer_blocks >= 1)
        {
            if (decoder_num_heads <= 0)
            {
                throw new ArgumentException($"{nameof(decoder_num_heads)} must be >0");
            }
            if (decoder_feed_forward_dim <= 0)
            {
                throw new ArgumentException($"{nameof(decoder_feed_forward_dim)} must be >0");
            }
            if (embedding_dim % decoder_num_heads != 0)
            {
                throw new ArgumentException($"{nameof(embedding_dim)} ({embedding_dim}) must be a multiple of {nameof(decoder_num_heads)} ({decoder_num_heads})");
            }
        }


        if (encoder_num_transformer_blocks <= 0 && decoder_num_transformer_blocks <= 0)
        {
            throw new ArgumentException($"{nameof(encoder_num_transformer_blocks)} and {nameof(decoder_num_transformer_blocks)} can not be 0 at the same time");
        }
    }
}
