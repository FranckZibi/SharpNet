using System;
using JetBrains.Annotations;
// ReSharper disable MemberCanBePrivate.Global
// ReSharper disable UnusedAutoPropertyAccessor.Global

namespace SharpNet.Models;

public class ModelDatasets
{
    public enum DatasetType
    {
        /// <summary>
        /// The first column of the dataset is the target, all remaining columns are the features.
        /// In this case Y_train_dataset_path & Y_validation_dataset_path should be null.
        /// </summary>
        LightGBMTrainingFormat,
    };

    public ModelDatasets([NotNull] string xTrainDatasetPath, [CanBeNull] string yTrainDatasetPath, [NotNull] string xValidationDatasetPath, [CanBeNull] string yValidationDatasetPath, [NotNull] string xTestDatasetPath, bool header, DatasetType type)
    {
        if (type == DatasetType.LightGBMTrainingFormat)
        {
            // in this format, xTrainDatasetPath and yTrainDatasetPath are equal
            if (!string.Equals(xTrainDatasetPath, yTrainDatasetPath))
            {
                throw new Exception($"{xTrainDatasetPath} != {yTrainDatasetPath}");
            }
            // in this format, xValidationDatasetPath and yValidationDatasetPath are equal
            if (!string.Equals(xValidationDatasetPath, yValidationDatasetPath))
            {
                throw new Exception($"{xValidationDatasetPath} != {yValidationDatasetPath}");
            }
        }

        X_train_dataset_path = xTrainDatasetPath;
        Y_train_dataset_path = yTrainDatasetPath;
        X_validation_dataset_path = xValidationDatasetPath;
        Y_validation_dataset_path = yValidationDatasetPath;
        X_test_dataset_path = xTestDatasetPath;
        Header = header;
        Type = type;
    }

    [NotNull]
    public string X_train_dataset_path { get; }
    [CanBeNull]
    public string Y_train_dataset_path { get; }
    [NotNull]
    public string X_validation_dataset_path { get; }
    [CanBeNull]
    public string Y_validation_dataset_path { get; }
    [NotNull]
    public string X_test_dataset_path { get; }

    /// <summary>
    /// true if all files in the dataset have header
    /// false if all files in the dataset have no headers
    /// </summary>
    public bool Header { get; }

    /// <summary>
    /// the kind of dataset contained
    /// </summary>
    public DatasetType Type { get; }

}