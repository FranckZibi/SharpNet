using System;
using System.Collections.Generic;
using System.Diagnostics;
using JetBrains.Annotations;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.Networks;
using SharpNet.Optimizers;

namespace SharpNet.Layers
{
    /// <summary>
    /// This layer can only be used as the second layer in a model (the first layer being the InputLayer).
    /// input shape :
    ///     (batchSize,  PrevLayer.n_x)
    /// output shape :
    ///     (batchSize,  PrevLayer.n_x, EmbeddingDim)
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
        /// same shape as 'Weights'
        /// </summary>
        [NotNull] private Tensor _weightGradients;
        /// <summary>
        /// Adam or SGD optimizer or Vanilla SGD
        /// </summary>
        #endregion
        [NotNull] private readonly Optimizer _optimizer;
        #endregion
        #region public fields and properties
        /// <summary>
        ///  Size of the vocabulary, i.e. maximum integer index + 1
        /// </summary>
        public int VocabularySize { get; }
        /// <summary>
        ///  Dimension of the dense embedding.
        /// </summary>
        public int EmbeddingDim { get; }
        /// <summary>
        /// regularization hyper parameter. 0 if no L2 regularization
        /// </summary>
        public double LambdaL2Regularization { get; }
        #endregion

        #region constructor
        public EmbeddingLayer(
            int vocabularySize,
            int embeddingDim,
            double lambdaL2Regularization,
            bool trainable, Network network, string layerName) : base(network, layerName)
        {
            VocabularySize = vocabularySize;
            EmbeddingDim = embeddingDim;
            LambdaL2Regularization = lambdaL2Regularization;

            Trainable = trainable;

            //trainable params
            _weights = GetFloatTensor(new[] { VocabularySize, EmbeddingDim });
            _weightGradients = GetFloatTensor(_weights.Shape);

            _optimizer = GetOptimizer(_weights.Shape, null);
            ResetParameters(false);
            //we set to 0 the first row of word embedding (for wordIndex = 0 which is not used)
            _weights.ElementSlice(0).ZeroMemory();
        }
        #endregion

        #region forward and backward propagation
        public override void ForwardPropagation(List<Tensor> allX, Tensor y, bool isTraining)
        {
            Debug.Assert(allX.Count == 1);
            var x = allX[0]; 
            Debug.Assert(x.Shape.Length == 2); // input 'x' tensor must be of shape (batchSize, maxWordsBySentence)
            Debug.Assert(y.Shape.Length == 3); // output 'y' tensor must be of shape (batchSize, maxWordsBySentence, EmbeddingDim)
            Debug.Assert(x.Shape[0] == y.Shape[0]); //same batch size
            Debug.Assert(x.Shape[1] == y.Shape[1]); //same word count by sentence
            //We compute y = x*Weights
            y.WordEmbeddingForwardPropagation(x, _weights);
        }
        public override void BackwardPropagation(List<Tensor> allX, Tensor y_NotUsed, Tensor dy, List<Tensor> allDx)
        {
            Debug.Assert(allX.Count == 1);
            Debug.Assert(y_NotUsed == null);

            var x = allX[0];
            int batchSize = dy.Shape[0];

            //we compute dW
            _weightGradients.WordEmbeddingBackwardPropagation(x, dy);

            //L2 regularization on dW
            if (UseL2Regularization)
            {
                var alpha = 2 * batchSize * (float)LambdaL2Regularization;
                _weightGradients.Update_Adding_Alpha_X(alpha, _weights);
            }
        }
        public override bool OutputNeededForBackwardPropagation => false;
        public override bool InputNeededForBackwardPropagation => true;
        #endregion

        #region parameters and gradients
        public override Tensor Weights => _weights;
        public override Tensor WeightGradients => _weightGradients;
        protected override Optimizer Optimizer => _optimizer;
        protected override bool HasParameters => true;
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
            _weights.RandomMatrixNormalDistribution(Rand, 0.0 /* mean */, Math.Sqrt(2.0 / PrevLayer.n_x) /*stdDev*/);

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
        public override IDictionary<string, CpuTensor<float>> GetParametersAsCpuFloatTensors(NetworkConfig.CompatibilityModeEnum originFramework)
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
                .Add(nameof(VocabularySize), VocabularySize)
                .Add(nameof(EmbeddingDim), EmbeddingDim)
                .Add(nameof(LambdaL2Regularization), LambdaL2Regularization)
                .ToString();
        }
        public static EmbeddingLayer Deserialize(IDictionary<string, object> serialized, Network network)
        {
            return new EmbeddingLayer(
                (int) serialized[nameof(VocabularySize)],
                (int) serialized[nameof(EmbeddingDim)],
                (double)serialized[nameof(LambdaL2Regularization)],
                (bool)serialized[nameof(Trainable)],
                network,
                (string)serialized[nameof(LayerName)]);
        }
        public override void AddToOtherNetwork(Network otherNetwork) { AddToOtherNetwork(otherNetwork, Deserialize); }
        #endregion

        public override int[] OutputShape(int batchSize)
        {
            return new[] { batchSize, PrevLayer.n_x, EmbeddingDim };
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
}
