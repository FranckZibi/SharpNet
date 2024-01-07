using SharpNet.Datasets;

namespace SharpNet.Networks.Transformers;

public abstract class TransformerDatasetSample : AbstractDatasetSample
{
    #region Hyperparameters
    public int vocab_size = 4;
    public int max_length = 3; // == timeSteps
    #endregion

    public override int[] X_Shape(int batchSize) => new[] { batchSize, max_length };
    public override int[] Y_Shape(int batchSize) => new[] { batchSize, vocab_size };
    public override int NumClass => vocab_size;

}