using System.Collections.Generic;
using System.Diagnostics;
using JetBrains.Annotations;
using SharpNet.Data;

namespace SharpNet.Optimizers;

public class VanillaSgdOrtho : Optimizer
{
    #region private fields
    // the orthogonal 'Q' matrix of shape (m, n)
    [CanBeNull] private readonly Tensor Q;
    //[CanBeNull] private readonly Tensor Identity_mm;
    [CanBeNull] private readonly Tensor buffer_mm;
    //[CanBeNull] private readonly Tensor buffer_mn_v1;
    //[CanBeNull] private readonly Tensor buffer_nn_v1;
    //  upper triangular matrix 'R' of shape (n, n)
    [CanBeNull] private readonly Tensor R;
    [CanBeNull] private readonly Tensor QRFactorization_buffer;
    private readonly TensorMemoryPool _memoryPool;

    #endregion

    public VanillaSgdOrtho(TensorMemoryPool memoryPool, int[] weightShape)
    {
        _memoryPool = memoryPool;
        Debug.Assert(weightShape.Length == 2);
        int m = weightShape[0];
        int n = weightShape[1];
        Debug.Assert(m >= n);
        Q = _memoryPool.GetFloatTensor(new[] { m, n });
        R = _memoryPool.GetFloatTensor(new[] { n, n });
        buffer_mm = _memoryPool.GetFloatTensor(new[] { m, m });
        //Identity_mm = _memoryPool.GetFloatTensor(new[] { m, m });
        //buffer_mn_v1 = _memoryPool.GetFloatTensor(new[] { m, n });
        //buffer_nn_v1 = _memoryPool.GetFloatTensor(new[] { n, n });
        //Identity_mm.SetIdentityMatrix();

        QRFactorization_buffer = _memoryPool.GetFloatTensor(new[] { Q.QRFactorization_FloatBufferLength() });
        ZeroMemory();
    }

    public override List<Tensor> EmbeddedTensors
    {
        get
        {
            var result = new List<Tensor> { Q, R, QRFactorization_buffer};
            result.RemoveAll(t => t == null);
            return result;
        }
    }

    public override bool IsOrthogonal => true;


    public override void UpdateWeights(double learningRate, double maxLearningRate, int batchSize, Tensor weights, Tensor weightGradients, Tensor bias, Tensor biasGradients)
    {
        Debug.Assert(weights.SameShape(weightGradients));
        Debug.Assert(bias == null || bias.SameShape(biasGradients));
        var ponderedLearningRate = PonderedLearning(learningRate, batchSize);
        /*
          W_new = 
            W 
            - 
            group['lr'].cuda()
            *
            (
               torch.mm(													(m,n)
                        tmp_I - torch.mm(W, W.t()) ,						(m, m)
                        dW													(m, n)
                       ) 
             + 0.5*torch.mm(												(m,n)
                        W, 													(m, n)
                        torch.mm(W.t(), dW) - torch.mm(dW.t(), W)			(n,n)
                       )
            )
        */

        var buffer_mn = Q;
        var buffer_nn = R;

        buffer_mm.SetIdentityMatrix();
        buffer_mm.Dot(weights, false, weights, true, -1.0f, 1.0f); // tmp_I - torch.mm(W, W.t())
        buffer_mn.Dot(buffer_mm, weightGradients); // torch.mm( tmp_I - torch.mm(W, W.t()) , dW )

        buffer_nn.Dot(weights, true, weightGradients, false, 1.0f, 0.0f); // torch.mm(W.t(), dW)
        buffer_nn.Dot(weightGradients, true, weights, false, -1.0f, 1.0f); //torch.mm(W.t(), dW) - torch.mm(dW.t(), W)
        buffer_mn.Dot(weights, false, buffer_nn, false, 0.5f, 1.0f); // torch.mm( tmp_I - torch.mm(W, W.t()) , dW )+ 0.5*torch.mm( W, torch.mm( tmp_I - torch.mm(W, W.t()) , dW ) )

        weights.Update_Adding_Alpha_X(-ponderedLearningRate, buffer_mn);
        weights.QRFactorization(Q,R, QRFactorization_buffer);
        //?D TODO update weights to make it unique
        Q.CopyTo(weights);

        /*
          W_new = W - group['lr'].cuda()*(torch.mm((tmp_I - torch.mm(W, W.t())),dW) + 0.5*torch.mm(W, (torch.mm(W.t(), dW) - torch.mm(dW.t(), W))))
                            Q, R = torch.qr(W_new)
                            D = torch.diag(torch.sign(torch.diag(R)))
                            Q = torch.mm(Q,D)

                            if n_feature_in <= n_feature_out:
                                p.data = Q
         */


        bias?.Update_Adding_Alpha_X(-ponderedLearningRate, biasGradients);
    }
    public override void Dispose()
    {
        if (_isDisposed)
        {
            return;
        }
        _isDisposed = true;
        base.Dispose();
        EmbeddedTensors.ForEach(t => _memoryPool?.FreeFloatTensor(t));
    }

    #region serialization
    public override string Serialize()
    {
        return new Serializer()
            .Add(nameof(Q), Q)
            .Add(nameof(R), R)
            .Add(nameof(QRFactorization_buffer), QRFactorization_buffer)
            .ToString();
    }

    public static Optimizer DeserializeVanillaSgdOrtho(IDictionary<string, object> serialized)
    {
        return serialized.ContainsKey(nameof(QRFactorization_buffer)) ? new VanillaSgdOrtho(serialized) : null;
    }
    private VanillaSgdOrtho(IDictionary<string, object> serialized)
    {
        serialized.TryGet(nameof(Q), out Q);
        serialized.TryGet(nameof(R), out R);
        serialized.TryGet(nameof(QRFactorization_buffer), out QRFactorization_buffer);
    }
    #endregion
}