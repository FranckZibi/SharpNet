using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using JetBrains.Annotations;
using log4net;
using Porter2StemmerStandard;
using SharpNet.CPU;
using SharpNet.Datasets;

namespace SharpNet.TextPreprocessing;

public static class TfIdfEncoding
{
    private static readonly IStemmer _stemmer = new EnglishPorter2Stemmer();

    private static readonly ILog Log = LogManager.GetLogger(typeof(DataSet));

    /// <summary>
    /// return the list of columns in 'df' that contains TfIdf encoding of column 'columnToEncode'
    /// </summary>
    /// <param name="df"></param>
    /// <param name="columnToEncode"></param>
    /// <returns></returns>
    private static string[] EncodedColumns(DataFrame df, string columnToEncode)
    {
        return df.Columns.Where(c => c.StartsWith(columnToEncode + TFIDF_COLUMN_NAME_KEY)).ToArray();
    }


    /// <summary>
    /// reduces the number of TfIdf features encoding the column 'columnToAdjustEncoding' in DataFrame 'df' to 'targetEmbeddingDim' columns,
    /// and return the new DataFrame.
    /// If the number of encoding columns for column 'columnToAdjustEncoding' is less than 'targetEmbeddingDim',
    /// then do nothing and returns the same the DataFrame
    /// </summary>
    /// <param name="df"></param>
    /// <param name="columnToAdjustEncoding"></param>
    /// <param name="targetEmbeddingDim"></param>
    /// <returns>a DataFrame with 'targetEmbeddingDim' (or less) columns used to encode 'columnToAdjustEncoding'</returns>
    public static DataFrame ReduceEncodingToTargetEmbeddingDim(DataFrame df, string columnToAdjustEncoding,
        int targetEmbeddingDim)
    {
        var existingEncodingColumns = EncodedColumns(df, columnToAdjustEncoding);
        if (existingEncodingColumns.Length <= targetEmbeddingDim)
        {
            Log.Debug(
                $"can't reduce encoding of column of {columnToAdjustEncoding} to {targetEmbeddingDim} because existing encoding is {existingEncodingColumns.Length}");
            return df;
        }

        return df.Drop(existingEncodingColumns.Skip(targetEmbeddingDim).ToArray());
    }

    public static IEnumerable<string> ColumnToRemoveToFitEmbedding(DataFrame df, string columnToAdjustEncoding,
        int targetEmbeddingDim, bool keepOriginalColumnNameWhenUsingEncoding)
    {
        var existingEncodingColumns = EncodedColumns(df, columnToAdjustEncoding).ToList();
        var toDrop = new List<string>();
        if (existingEncodingColumns.Count > targetEmbeddingDim)
        {
            toDrop = existingEncodingColumns.Skip(targetEmbeddingDim).ToList();
        }

        if (existingEncodingColumns.Count > toDrop.Count && !keepOriginalColumnNameWhenUsingEncoding)
        {
            toDrop.Add(columnToAdjustEncoding);
        }

        return toDrop;
    }


    public enum TfIdfEncoding_norm
    {
        None,
        L1,
        L2,
        Standardization // we scale the embedding with 0 mean and 1 as standard deviation (= volatility). This is not a norm
    };
    public static List<DataFrame> Encode(IList<DataFrame> dfs, string columnToEncode, int embedding_dim, bool keepEncodedColumnName = false, bool reduceEmbeddingDimIfNeeded = false, TfIdfEncoding_norm norm = TfIdfEncoding_norm.L2, bool scikitLearnCompatibilityMode = false)
    {
        var documents = new List<string>();
        foreach (var df in dfs)
        {
            documents.AddRange(df?.StringColumnContent(columnToEncode) ?? Array.Empty<string>());
        }

        var documents_tfidf_encoded = Encode(documents, embedding_dim, columnToEncode, reduceEmbeddingDimIfNeeded, norm, scikitLearnCompatibilityMode);
        var dfs_encoded = new List<DataFrame>();
        var startRowIndex = 0;
        for (var index = 0; index < dfs.Count; index++)
        {
            var df = dfs[index];
            if (df == null)
            {
                dfs_encoded.Add(null);
                continue;
            }

            if (!keepEncodedColumnName)
            {
                df = df.Drop(columnToEncode);
            }

            var df_tfidf_encoded = documents_tfidf_encoded.RowSlice(startRowIndex, df.Shape[0], false);
            dfs_encoded.Add(DataFrame.MergeHorizontally(df, df_tfidf_encoded));
            startRowIndex += df.Shape[0];
        }

        return dfs_encoded;
    }



    public static DataFrame Encode(IList<string> documents, int embedding_dim, [NotNull] string columnNameToEncode, bool reduceEmbeddingDimIfNeeded = false, TfIdfEncoding_norm norm = TfIdfEncoding_norm.L2, bool scikitLearnCompatibilityMode = false)
    {
        var tokenizer = new Tokenizer(oovToken: null, lowerCase: true, stemmer: _stemmer);
       
        tokenizer.FitOnTexts(documents);
        var training_sequences = tokenizer.TextsToSequences(documents);
        if (reduceEmbeddingDimIfNeeded && tokenizer.DistinctWords < embedding_dim)
        {
            embedding_dim = Math.Max(1, tokenizer.DistinctWords);
        }

        Log.Info($"number of distinct words for {columnNameToEncode}: {tokenizer.DistinctWords}");

        var tfIdf = new float[documents.Count * embedding_dim];
        // documentsContainingWord[wordIdx] : number of distinct documents containing the word at index 'wordIdx'
        var documentsContainingWord = new int[embedding_dim];

        //first step: we compute the Text Frequency (tf)
        //  tfIdf[documentId * embedding_dim + wordIdx] : number of time the word at 'wordIdx' appears in document 'documentId'
        for (int documentId = 0; documentId < documents.Count; ++documentId)
        {
            var wordIdxStartingAt1S = training_sequences[documentId];
            foreach (var wordIdx_starting_at_1 in wordIdxStartingAt1S)
            {
                // the first id start at 1, we want it to start at 0
                Debug.Assert(wordIdx_starting_at_1 >= 1);
                int wordIdx = wordIdx_starting_at_1 - 1;
                if (wordIdx >= embedding_dim | wordIdx < 0)
                {
                    continue;
                }

                if (tfIdf[documentId * embedding_dim + wordIdx] == 0)
                {
                    ++documentsContainingWord[wordIdx];
                }
                tfIdf[documentId * embedding_dim + wordIdx] += 1f/ wordIdxStartingAt1S.Count;
            }
        }

        //second step:
        //  tfIdf[documentId * columns + wordIdx] : tf*idf of word at 'wordIdx' for document 'documentId'
        for (int documentId = 0; documentId < documents.Count; ++documentId)
        {
            for (int wordIdx = 0; wordIdx < embedding_dim; ++wordIdx)
            {
                //number of time the word at 'wordIdx' appears in document 'documentId'
                var tf = tfIdf[documentId * embedding_dim + wordIdx];
                if (tf == 0)
                {
                    continue;
                }
                if (scikitLearnCompatibilityMode)
                {
                    float df = (1f+documentsContainingWord[wordIdx]) / (1f+documents.Count);
                    float idf = MathF.Log(1 / df)+1;
                    tfIdf[documentId * embedding_dim + wordIdx] = tf * idf;
                }
                else
                {
                    // % of document containing (at least one time) the word at 'wordIdx'
                    float df = ((float)documentsContainingWord[wordIdx]) / documents.Count;
                    float idf = MathF.Log(1 / df);
                    tfIdf[documentId * embedding_dim + wordIdx] = tf * idf;
                }
            }
        }

        if (norm != TfIdfEncoding_norm.None)
        {
            for (int documentId = 0; documentId < documents.Count; ++documentId)
            {
                double sumForRow = 0.0;         // for Standardization
                double absSumForRow = 0.0;      // for L1 norm
                double sumSquareForRow = 0.0;   // for L2 norm and Standardization
                for (int wordIdx = 0; wordIdx < embedding_dim; ++wordIdx)
                {
                    var val = tfIdf[documentId * embedding_dim + wordIdx];
                    sumForRow += val;
                    absSumForRow += Math.Abs(val);
                    sumSquareForRow +=(val * val);
                }

                float multiplier;
                float toAdd;
                switch (norm)
                {
                    case TfIdfEncoding_norm.L1:
                        multiplier = absSumForRow>0?(1 / (float)absSumForRow):0;
                        toAdd = 0.0f;
                        break;
                    case TfIdfEncoding_norm.L2:
                        multiplier = sumSquareForRow>0?(1 / (float)Math.Sqrt(sumSquareForRow)):0;
                        toAdd = 0.0f;
                        break;
                    case TfIdfEncoding_norm.Standardization:
                        var mean = (float)sumForRow / embedding_dim;
                        var variance = Math.Abs(sumSquareForRow - embedding_dim * mean * mean) / embedding_dim;
                        var stdDev = (float)Math.Sqrt(variance);
                        multiplier = (stdDev>0)? (1 / stdDev):0;
                        toAdd = (stdDev > 0)?(-mean/ stdDev):0;
                        break;
                    default:
                        throw new NotSupportedException($"invalid norm {norm}");
                }
                for (int wordIdx = 0; wordIdx < embedding_dim; ++wordIdx)
                {
                    var index = documentId * embedding_dim + wordIdx;
                    tfIdf[index] = multiplier* tfIdf[index]+toAdd;
                }
            }
        }

        //we create the column names
        var sequenceToWords = tokenizer.SequenceToWords(Enumerable.Range(1, embedding_dim)).Select(x => x ?? "OOV")
            .ToList();
        List<string> columns = new();
        for (int i = 1; i <= embedding_dim; ++i)
        {
            var tfidfColumnName = columnNameToEncode + TFIDF_COLUMN_NAME_KEY + sequenceToWords[i - 1];
            columns.Add(tfidfColumnName);
        }

        var tensor = CpuTensor<float>.New(tfIdf, columns.Count);
        return DataFrame.New(tensor, columns);
    }

    private const string TFIDF_COLUMN_NAME_KEY = "_tfidf_";

}
