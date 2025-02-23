using SharpNet.Datasets;

namespace SharpNet.Networks.Transformers;

public abstract class TransformerDatasetSample : AbstractDatasetSample
{
    #region HyperParameters
    public int num_embeddings = 4;
    public int max_length = 3; // == timeSteps
    #endregion

    public override int[] X_Shape(int batchSize) => new[] { batchSize, max_length };
    public override int[] Y_Shape(int batchSize) => new[] { batchSize, num_embeddings };
    //public override int[] Y_Shape(int batchSize) => new[] { batchSize, max_length , num_embeddings };
    public override int NumClass => num_embeddings;

}