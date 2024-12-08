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
/// Root Mean Square Layer Normalization, see https://arxiv.org/abs/1910.07467
/// </summary>
public sealed class RMSNorm : Layer
{
    #region Private fields
    private readonly double _epsilon;
    //we'll normalize over the last 'D' dimension of the input tensor
    // for example:
    //  if D == 3 and input shape is [N,C,H,W]:
    //          we'll normalize over [C,H,W]
    //          the input tensor will be treated as a 2D matrix (N, C*H*W)
    //  if D == 2 and input shape is [N,C,H]:
    //          we'll normalize over [C,H]
    //          the input tensor will be treated as a 2D matrix (N, C*H)
    //  if D == 1 and input shape is [N,C]:
    //          we'll normalize over the last dimension [C]
    //          the input tensor will be treated as a 2D matrix (N, C)
    // 
    private readonly int _last_D_dimension;
    #region trainable parameters
    /// <summary>
    /// Scale (= gammas) Tensor
    /// </summary>
    private Tensor _gammas;
    #endregion

    #region buffers
    [CanBeNull] private Tensor _mean_squares_buffer;
    #endregion

    #region gradients
    [NotNull] private Tensor _gammasGradients;         // same shape as '_gammas"
    #endregion
    /// <summary>
    /// Adam or SGD optimizer or Vanilla SGD
    /// </summary>
    [NotNull] private readonly Optimizer _optimizer;
    #endregion

    // to check with actual value used in PyTorch
    public const float DEFAULT_EPSILON = 1.2f * 1e-7f;

    public RMSNorm(int last_D_dimension, double epsilon, bool trainable, Network network, string layerName, int prevLayerIndex) : base(network, prevLayerIndex == -1 ? new[] { network.LastLayerIndex } : new[] { prevLayerIndex }, layerName)
    {
        _last_D_dimension = last_D_dimension;
        _epsilon = epsilon;
        Trainable = trainable;

        var scaleAndBiasShape = ScaleAndBiasShape();

        //trainable parameters 
        _gammas = GetFloatTensor(scaleAndBiasShape);

        //gradients
        _gammasGradients = GetFloatTensor(scaleAndBiasShape);

        _optimizer = Sample.GetOptimizer(_gammas.Shape, null, MemoryPool);

        //no need to reset optimizer weights: it has just been done above
        ResetParameters(false);
    }

    #region forward and backward propagation
    public override void ForwardPropagation(List<Tensor> allX, Tensor y, bool isTraining)
    {
        var x = allX[0];
        Debug.Assert(allX.Count == 1);
        int reshape_cols = _gammas.Count;
        if (x.Count % reshape_cols != 0)
        {
            throw new Exception("RMSNorm: input tensor count (" + x.Count + ") is not a multiple of the number of gammas (" + reshape_cols + ")");
        }
        int reshape_rows = allX[0].Count / reshape_cols;

        // we compute the sum of squares
        GetFloatTensor(ref _mean_squares_buffer, new[] { 1, reshape_rows });
        x.Compute_Mean_Squares_Buffer(_mean_squares_buffer);
        x.RMSNormalization(y, _gammas, _mean_squares_buffer, (float)_epsilon);
    }

    public override void BackwardPropagation(List<Tensor> allX, Tensor y_NotUsed, Tensor dy, List<Tensor> allDx)
    {
        var dx = allDx[0];
        var x = allX[0];

        //we compute '_gammasGradients'
        //hat_x = x / np.sqrt(variance + eps)
        //dgamma = np.sum(dy * hat_x, axis = 0)
        var hat_x = dx;
        x.CopyTo(hat_x);
        hat_x.RMSStandardizeInPlace(_mean_squares_buffer, (float)_epsilon);
        hat_x.Update_Multiply_By_x(dy);
        hat_x.numpy_sum(_gammasGradients, 0);

        var dmean_squares_row = GetFloatTensor(_mean_squares_buffer.Shape);
        //we compute 'dx'
        x.RMSNormalizationBackward(dy, dx, _gammas, _mean_squares_buffer, (float)_epsilon, dmean_squares_row);

        FreeFloatTensor(dmean_squares_row);
    }
    public override bool OutputNeededForBackwardPropagation => false;
    #endregion

    #region parameters and gradients
    public override Tensor Weights => _gammas;
    public override Tensor Bias => null;
    public override Tensor WeightGradients => _gammasGradients;
    public override Tensor BiasGradients => null;
    protected override Optimizer Optimizer => _optimizer;
    public override List<Tuple<Tensor, string>> Parameters
    {
        get
        {
            var result = new List<Tuple<Tensor, string>>
                         {
                             Tuple.Create(_gammas, GammasDatasetPath),
                         };
            return result;
        }
    }

    public override void ResetParameters(bool resetAlsoOptimizerWeights = true)
    {
        //trainable params
        _gammas.SetValue(1);
        if (resetAlsoOptimizerWeights)
        {
            _optimizer.ZeroMemory();
        }
    }

    #region Multi GPU Support
    public override void ReplaceParameters(List<Tensor> newParameters)
    {
        Debug.Assert(newParameters.Count == 2);
        FreeFloatTensor(ref _gammas);
        _gammas = newParameters[0];
    }
    public override void ReplaceGradients(List<Tensor> newGradients)
    {
        FreeFloatTensor(ref _gammasGradients);
        _gammasGradients = newGradients[0];
    }
    #endregion

    public override void LoadParameters(IDictionary<string, Tensor> h5FileDataset, NetworkSample.CompatibilityModeEnum originFramework)
    {
        foreach (var layerParameters in Parameters)
        {
            var parameterId = layerParameters.Item2;
            if (h5FileDataset.ContainsKey(parameterId))
            {
                h5FileDataset[parameterId].CopyTo(layerParameters.Item1);
            }
        }
    }
    public override IDictionary<string, CpuTensor<float>> GetParametersAsCpuFloatTensors(NetworkSample.CompatibilityModeEnum originFramework)
    {
        var result = new Dictionary<string, CpuTensor<float>>();
        result[GammasDatasetPath] = _gammas.ToCpuFloat();
        return result;
    }


    private string GammasDatasetPath => DatasetNameToDatasetPath("gamma:0");
    #endregion

    #region serialization
    public override string Serialize()
    {
        return RootSerializer()
            .Add(nameof(_last_D_dimension), _last_D_dimension)
            .Add(nameof(_epsilon), _epsilon)
            .ToString();
    }
    public static RMSNorm Deserialize(IDictionary<string, object> serialized, Network network)
    {
        int prevLayerIndex = ((int[])serialized[nameof(PreviousLayerIndexes)])[0];
        return new RMSNorm(
            (int)serialized[nameof(_last_D_dimension)],
            (double)serialized[nameof(_epsilon)],
            (bool)serialized[nameof(Trainable)],
            network,
            (string)serialized[nameof(LayerName)],
            prevLayerIndex);
    }
    public override void AddToOtherNetwork(Network otherNetwork) { AddToOtherNetwork(otherNetwork, Deserialize); }
    #endregion

    #region PyTorch support
    //see: https://pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html
    public override void ToPytorchModule(List<string> constructorLines, List<string> forwardLines)
    {
        var input_shape = PreviousLayers.Count == 0 ? new[] { -1, -1, -1, -1 } : PreviousLayers[0].OutputShape(666);
        var normalized_shape = Utils.ShapeToString(input_shape.Skip(input_shape.Length - _last_D_dimension).ToArray());
        constructorLines.Add("self." + LayerName + " = torch.nn.RMSNorm(normalized_shape=" + normalized_shape + ", eps=" + _epsilon + ")");
        UpdateForwardLines(forwardLines);
    }
    #endregion


    public override string ToString()
    {
        var result = LayerName + ": " + ShapeChangeDescription();
        result += " (" + TotalParams + " neurons)";
        return result;
    }

    protected override List<Tensor> EmbeddedTensors(bool includeOptimizeTensors)
    {
        var result = base.EmbeddedTensors(includeOptimizeTensors);
        result.AddRange(new[] { _mean_squares_buffer});
        result.RemoveAll(t => t == null);
        return result;
    }



    protected override string ComputeLayerName()
    {
        return "RMSNorm";
    }

    private int[] ScaleAndBiasShape()
    {
        var res = OutputShape(1);
        int cols = Utils.Product(res.Skip(res.Length - _last_D_dimension).ToArray());
        return new[] { 1, cols };
    }
}