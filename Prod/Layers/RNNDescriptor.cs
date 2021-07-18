using System;
using SharpNet.GPU;

namespace SharpNet.Layers
{
    public class RNNDescriptor
    {
        #region fields
        public cudnnRNNAlgo_t algo { get; }
        public cudnnRNNMode_t cellMode { get; }
        public cudnnRNNBiasMode_t biasMode { get; }
        public cudnnDirectionMode_t dirMode { get; }
        public cudnnRNNInputMode_t inputMode { get; }
        public cudnnDataType_t dataType { get; }
        public cudnnDataType_t mathPrec { get; }
        public cudnnMathType_t mathType { get; }
        public int inputSize { get; }
        public int hiddenSize { get; }
        public int projSize { get; }
        public int numLayers { get; }
        public double dropoutRate { get; }
        public uint auxFlags { get; }
        #endregion

        public RNNDescriptor(cudnnRNNAlgo_t algo, cudnnRNNMode_t cellMode, cudnnRNNBiasMode_t biasMode, cudnnDirectionMode_t dirMode, cudnnRNNInputMode_t inputMode, cudnnDataType_t dataType, cudnnDataType_t mathPrec, cudnnMathType_t mathType, int inputSize, int hiddenSize, int projSize, int numLayers, double dropoutRate, uint auxFlags)
        {
            this.algo = algo;
            this.cellMode = cellMode;
            this.biasMode = biasMode;
            this.dirMode = dirMode;
            this.inputMode = inputMode;
            this.dataType = dataType;
            this.mathPrec = mathPrec;
            this.mathType = mathType;
            this.inputSize = inputSize;
            this.hiddenSize = hiddenSize;
            this.projSize = projSize;
            this.numLayers = numLayers;
            this.dropoutRate = dropoutRate;
            this.auxFlags = auxFlags;
        }

        public int[] YRnnData_Shape(int timeSteps, int batchSize)
        {
            int K = (dirMode == cudnnDirectionMode_t.CUDNN_BIDIRECTIONAL) ? 2 : 1;
            return new[] {batchSize, timeSteps, K * hiddenSize };
        }

        public int[] Y_Shape(bool returnSequences, int timeSteps, int batchSize)
        {
            int K = (dirMode == cudnnDirectionMode_t.CUDNN_BIDIRECTIONAL) ? 2 : 1;
            return returnSequences
                ?new[] { batchSize, timeSteps, K * hiddenSize }
                :new[] { batchSize, K * hiddenSize };
        }


        /// <summary>
        /// total number of pseudo layers.
        /// see: https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnGetRNNWeightParams
        /// </summary>
        public int PseudoLayersCount
        {
            get
            {
                int K = (dirMode == cudnnDirectionMode_t.CUDNN_BIDIRECTIONAL) ? 2 : 1;
                return K * numLayers;
            }
        }



        /// <summary>
        /// total number of weight matrices in each pseudo layers:
        ///     the first half is associated with weight matrices (or bias vectors) used in conjunction with the input from the previous pseudo layer
        ///     the second half is associated with weight matrices (or bias vectors) used in conjunction with the hidden state from the previous time step
        /// see: https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnGetRNNWeightParams
        /// </summary>
        public int LinLayerIDCount
        {
            get
            {
                switch (cellMode)
                {
                    case cudnnRNNMode_t.CUDNN_RNN_RELU: return 2*1;
                    case cudnnRNNMode_t.CUDNN_RNN_TANH: return 2*1;
                    case cudnnRNNMode_t.CUDNN_LSTM: return 2*4;
                    case cudnnRNNMode_t.CUDNN_GRU: return 2*3;
                    default:
                        throw new NotImplementedException(nameof(LinLayerIDCount) + " " + cellMode);
                }
            }
        }

        public override bool Equals(object obj)
        {
            if (!(obj is RNNDescriptor))
            {
                return false;
            }
            var other = (RNNDescriptor)obj;
            return algo == other.algo && cellMode == other.cellMode && biasMode == other.biasMode && dirMode == other.dirMode && inputMode == other.inputMode && dataType == other.dataType && mathPrec == other.mathPrec && mathType == other.mathType && inputSize == other.inputSize && hiddenSize == other.hiddenSize && projSize == other.projSize && numLayers == other.numLayers && dropoutRate.Equals(other.dropoutRate) && auxFlags == other.auxFlags;
        }
        public override int GetHashCode()
        {
            unchecked
            {
                var hashCode = (int)algo;
                hashCode = (hashCode * 397) ^ (int)cellMode;
                hashCode = (hashCode * 397) ^ (int)biasMode;
                hashCode = (hashCode * 397) ^ (int)dirMode;
                hashCode = (hashCode * 397) ^ (int)inputMode;
                hashCode = (hashCode * 397) ^ (int)dataType;
                hashCode = (hashCode * 397) ^ (int)mathPrec;
                hashCode = (hashCode * 397) ^ (int)mathType;
                hashCode = (hashCode * 397) ^ inputSize;
                hashCode = (hashCode * 397) ^ hiddenSize;
                hashCode = (hashCode * 397) ^ projSize;
                hashCode = (hashCode * 397) ^ numLayers;
                hashCode = (hashCode * 397) ^ dropoutRate.GetHashCode();
                hashCode = (hashCode * 397) ^ (int)auxFlags;
                return hashCode;
            }
        }
    }
}
