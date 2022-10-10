namespace SharpNet.Datasets;

public enum ResizeStrategyEnum
{
    /// <summary>
    /// we do n ot resize the image from disk to the target size for training/inference
    /// we expect them to be the same
    /// an exception is thrown if it is not the case
    /// </summary>
    None,

    /// <summary>
    /// we'll simply resize the image from disk to the target size for training/inference
    /// without keeping the same proportion.
    /// It means that the picture can be distorted to fit the target size
    /// </summary>
    ResizeToTargetSize,

    /// <summary>
    /// We'll resize the image so that it will have exactly the same width as the size fo the training/inference tensor
    /// We'll keep the same proportion as in the original image (no distortion)
    /// </summary>
    // ReSharper disable once UnusedMember.Global
    ResizeToWidthSizeKeepingSameProportion,

    /// <summary>
    /// We'll resize the image so that it will have exactly the same height as the size fo the training/inference tensor
    /// We'll keep the same proportion as in the original image (no distortion)
    /// </summary>
    // ReSharper disable once UnusedMember.Global
    ResizeToHeightSizeKeepingSameProportion,

    /// <summary>
    /// We'll take the biggest crop in the original image and resize this crop to match exactly the size fo the training/inference tensor
    /// We'll keep the same proportion as in the original image (no distortion)
    /// </summary>
    // ReSharper disable once UnusedMember.Global
    BiggestCropInOriginalImageToKeepSameProportion
}