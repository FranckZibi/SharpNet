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
/// Layer Normalization, see https://arxiv.org/abs/1607.06450
/// </summary>
public sealed class LayerNorm : Layer
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
    /// <summary>
    /// Bias (= betas = offset) Tensor
    /// </summary>
    private Tensor _betas;                           // same shape as '_gammas"
    #endregion

    #region buffers
    [NotNull] private Tensor _mean_buffer;
    [NotNull] private Tensor _variance_buffer;
    #endregion

    #region gradients
    [NotNull] private Tensor _gammasGradients;         // same shape as '_gammas"
    [CanBeNull] private Tensor _betasGradients;        // same shape as '_gammas"
    #endregion
    /// <summary>
    /// Adam or SGD optimizer or Vanilla SGD
    /// </summary>
    [NotNull] private readonly Optimizer _optimizer;
    #endregion

    public const float DEFAULT_EPSILON = 1e-5f;

    public LayerNorm(int last_D_dimension, double epsilon, bool trainable, Network network, string layerName, int prevLayerIndex) : base(network, prevLayerIndex == -1 ? new[]{network.LastLayerIndex}:new[]{ prevLayerIndex }, layerName)
    {
        _last_D_dimension = last_D_dimension;
        _epsilon = epsilon;
        Trainable = trainable;

        var scaleAndBiasShape = ScaleAndBiasShape();

        //trainable parameters 
        _gammas = GetFloatTensor(scaleAndBiasShape);
        _betas = GetFloatTensor(scaleAndBiasShape);

        //gradients
        _gammasGradients = GetFloatTensor(scaleAndBiasShape);
        _betasGradients = GetFloatTensor(scaleAndBiasShape);

        _optimizer = Sample.GetOptimizer(_gammas.Shape, _betas.Shape, MemoryPool);

        //no need to reset optimizer weights: it has just been done above
        ResetParameters(false);

        ////We disable bias for the previous layers
        //var nbDisabledWeights = PreviousLayers.Select(l => l.DisableBias()).Sum();
        //if (nbDisabledWeights != 0)
        //{
        //    Log(nbDisabledWeights + " weights (bias) disabled thanks to LayerNormalization layer " + LayerName);
        //}
    }

    #region forward and backward propagation
    public override void ForwardPropagation(List<Tensor> allX, Tensor y, bool isTraining)
    {
        var x = allX[0];
        Debug.Assert(allX.Count == 1);
        int reshape_cols = _gammas.Count;
        if (x.Count % reshape_cols != 0)
        {
            throw new Exception("LayerNorm: input tensor count (" + x.Count + ") is not a multiple of the number of gammas (" + reshape_cols + ")");
        }
        int reshape_rows = allX[0].Count / reshape_cols;

        // we compute the mean / variance 
        GetFloatTensor(ref _mean_buffer, new[] { 1, reshape_rows });
        GetFloatTensor(ref _variance_buffer, _mean_buffer.Shape);
        x.Compute_Row_Mean_Variance(_mean_buffer, _variance_buffer, false);
        x.LayerNormalization(y, _gammas, _betas, _mean_buffer, _variance_buffer, (float)_epsilon);
    }

    public override void BackwardPropagation(List<Tensor> allX, Tensor y_NotUsed, Tensor dy, List<Tensor> allDx)
    {
        var dx = allDx[0];
        var x = allX[0];

        //we compute '_betasGradients'
        //_betasGradients = 	np.sum(dy, axis = 0)
        dy.numpy_sum(_betasGradients, 0);

        //we compute '_gammasGradients'
        //hat_x = (x - mean) / np.sqrt(variance + eps)
        //dgamma = np.sum(dy * hat_x, axis = 0)
        var hat_x = dx;
        x.CopyTo(hat_x);
        hat_x.StandardizeInPlace(_mean_buffer, _variance_buffer, 1, (float)_epsilon);
        hat_x.Update_Multiply_By_x(dy);
        hat_x.numpy_sum(_gammasGradients, 0);

        var dmean_row = GetFloatTensor(_mean_buffer.Shape);
        var dvariance_row = GetFloatTensor(_mean_buffer.Shape);
        //we compute 'dx'
        x.LayerNormalizationBackward(dy, dx, _gammas, _mean_buffer, _variance_buffer, (float)_epsilon, dmean_row, dvariance_row);

        FreeFloatTensor(dmean_row);
        FreeFloatTensor(dvariance_row);
    }
    public override bool OutputNeededForBackwardPropagation => false;
    #endregion

    #region parameters and gradients
    public override Tensor Weights => _gammas;
    public override Tensor Bias => _betas;
    public override Tensor WeightGradients => _gammasGradients;
    public override Tensor BiasGradients => _betasGradients;
    protected override Optimizer Optimizer => _optimizer;
    public override List<Tuple<Tensor, string>> Parameters
    {
        get
        {
            var result = new List<Tuple<Tensor, string>>
            {
                Tuple.Create(_gammas, GammasDatasetPath),
                Tuple.Create(_betas, BetasDatasetPath),
            };
            return result;
        }
    }

    public override void ResetParameters(bool resetAlsoOptimizerWeights = true)
    {
        //trainable params
        _gammas.SetValue(1);
        _betas.ZeroMemory();
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
        FreeFloatTensor(ref _betas);
        _betas = newParameters[1];
    }
    public override void ReplaceGradients(List<Tensor> newGradients)
    {
        FreeFloatTensor(ref _gammasGradients);
        _gammasGradients = newGradients[0];
        if (_betasGradients != null)
        {
            Debug.Assert(newGradients.Count == 2);
            FreeFloatTensor(ref _betasGradients);
            _betasGradients = newGradients[1];
        }
        else
        {
            Debug.Assert(newGradients.Count == 1);
        }
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
        result[BetasDatasetPath] = _betas.ToCpuFloat();
        return result;
    }


    private string GammasDatasetPath => DatasetNameToDatasetPath("gamma:0");
    private string BetasDatasetPath => DatasetNameToDatasetPath("beta:0");
    #endregion

    #region serialization
    public override string Serialize()
    {
        return RootSerializer()
            .Add(nameof(_last_D_dimension), _last_D_dimension)
            .Add(nameof(_epsilon), _epsilon)
            .ToString();
    }
    public static LayerNorm Deserialize(IDictionary<string, object> serialized, Network network)
    {
        int prevLayerIndex = ((int[])serialized[nameof(PreviousLayerIndexes)])[0];
        return new LayerNorm(
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
    //see: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    public override void ToPytorchModule(List<string> constructorLines, List<string> forwardLines)
    {
        var input_shape = PreviousLayers.Count == 0 ? new[] { -1, -1, -1, -1 } : PreviousLayers[0].OutputShape(666);
        var normalized_shape = Utils.ShapeToString(input_shape.Skip(input_shape.Length - _last_D_dimension).ToArray());
        constructorLines.Add("self." + LayerName + " = torch.nn.LayerNorm(normalized_shape=" + normalized_shape + ", eps=" + _epsilon + ")");
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
        result.AddRange(new[] { _mean_buffer, _variance_buffer});
        result.RemoveAll(t => t == null);
        return result;
    }

    

    protected override string ComputeLayerName()
    {
        return "layer_normalization";
    }

    private int[] ScaleAndBiasShape()
    {
        var res = OutputShape(1);
        int cols = Utils.Product(res.Skip(res.Length - _last_D_dimension).ToArray());
        return new[] { 1, cols };
    }
}