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
        int targetEmbeddingDim, bool keepOrginalColumnNameWhenUsingEncoding)
    {
        var existingEncodingColumns = EncodedColumns(df, columnToAdjustEncoding).ToList();
        var toDrop = new List<string>();
        if (existingEncodingColumns.Count > targetEmbeddingDim)
        {
            toDrop = existingEncodingColumns.Skip(targetEmbeddingDim).ToList();
        }

        if (existingEncodingColumns.Count > toDrop.Count && !keepOrginalColumnNameWhenUsingEncoding)
        {
            toDrop.Add(columnToAdjustEncoding);
        }

        return toDrop;
    }


    public enum TfIdfEncoding_norm
    {
        None,
        L1,
        L2
    };
    public static List<DataFrame> Encode(IList<DataFrame> dfs, string columnToEncode, int embeddingDim,
        bool keepEncodedColumnName = false, bool addTokenNameAsColumnNameSuffix = false,
        bool reduceEmbeddingDimIfNeeded = false, TfIdfEncoding_norm norm = TfIdfEncoding_norm.L2, bool scikitLearnCompatibilityMode = false)
    {
        var documents = new List<string>();
        foreach (var df in dfs)
        {
            documents.AddRange(df?.StringColumnContent(columnToEncode) ?? Array.Empty<string>());
        }

        var documents_tfidf_encoded = Encode(documents, embeddingDim, columnToEncode, addTokenNameAsColumnNameSuffix,
            reduceEmbeddingDimIfNeeded, norm, scikitLearnCompatibilityMode);
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



    public static DataFrame Encode(IList<string> documents, int embeddingDim, [NotNull] string columnNameToEncode,
        bool addTokenNameAsColumnNameSuffix = false, bool reduceEmbeddingDimIfNeeded = false,
        TfIdfEncoding_norm norm = TfIdfEncoding_norm.L2, bool scikitLearnCompatibilityMode = false)
    {
        var tokenizer = new Tokenizer(oovToken: null, lowerCase: true, stemmer: _stemmer);
       
        tokenizer.FitOnTexts(documents);
        var training_sequences = tokenizer.TextsToSequences(documents);
        if (reduceEmbeddingDimIfNeeded && tokenizer.DistinctWords < embeddingDim)
        {
            embeddingDim = Math.Max(1, tokenizer.DistinctWords);
        }

        Log.Info($"number of distinct words for {columnNameToEncode}: {tokenizer.DistinctWords}");

        var tfIdf = new float[documents.Count * embeddingDim];
        // documentsContainingWord[wordIdx] : number of distinct documents containing the word at index 'wordIdx'
        var documentsContainingWord = new int[embeddingDim];

        //first step: we compute the Text Frequency (tf)
        //  tfIdf[documentId * embeddingDim + wordIdx] : number of time the word at 'wordIdx' appears in document 'documentId'
        for (int documentId = 0; documentId < documents.Count; ++documentId)
        {
            var wordIdxStartingAt1S = training_sequences[documentId];
            foreach (var wordIdx_starting_at_1 in wordIdxStartingAt1S)
            {
                // the first id start at 1, we want it to start at 0
                Debug.Assert(wordIdx_starting_at_1 >= 1);
                int wordIdx = wordIdx_starting_at_1 - 1;
                if (wordIdx >= embeddingDim | wordIdx < 0)
                {
                    continue;
                }

                if (tfIdf[documentId * embeddingDim + wordIdx] == 0)
                {
                    ++documentsContainingWord[wordIdx];
                }
                tfIdf[documentId * embeddingDim + wordIdx] += 1f/ wordIdxStartingAt1S.Count;
            }
        }

        //second step:
        //  tfIdf[documentId * columns + wordIdx] : tf*idf of word at 'wordIdx' for document 'documentId'
        for (int documentId = 0; documentId < documents.Count; ++documentId)
        {
            for (int wordIdx = 0; wordIdx < embeddingDim; ++wordIdx)
            {
                //number of time the word at 'wordIdx' appears in document 'documentId'
                var tf = tfIdf[documentId * embeddingDim + wordIdx];
                if (tf == 0)
                {
                    continue;
                }
                if (scikitLearnCompatibilityMode)
                {
                    float df = (1f+documentsContainingWord[wordIdx]) / (1f+documents.Count);
                    float idf = (float)Math.Log(1 / df)+1;
                    tfIdf[documentId * embeddingDim + wordIdx] = tf * idf;
                }
                else
                {
                    // % of document containing (at least one time) the word at 'wordIdx'
                    float df = ((float)documentsContainingWord[wordIdx]) / documents.Count;
                    float idf = (float)Math.Log(1 / df);
                    tfIdf[documentId * embeddingDim + wordIdx] = tf * idf;
                }
            }
        }

        if (norm != TfIdfEncoding_norm.None)
        {
            for (int documentId = 0; documentId < documents.Count; ++documentId)
            {
                double absSumForRow = 0.0;      // for L1 norm
                double sumSquareForRow = 0.0;   // for L2 norm
                for (int wordIdx = 0; wordIdx < embeddingDim; ++wordIdx)
                {
                    var val = tfIdf[documentId * embeddingDim + wordIdx];
                    absSumForRow += Math.Abs(val);
                    sumSquareForRow +=(val * val);
                }
                if (absSumForRow > 0) //if some non zero elements found
                {
                    var multiplier =  (norm == TfIdfEncoding_norm.L1) ? (1 / (float)absSumForRow) : (1 / (float)Math.Sqrt(sumSquareForRow));
                    for (int wordIdx = 0; wordIdx < embeddingDim; ++wordIdx)
                    {
                        tfIdf[documentId * embeddingDim + wordIdx] *= multiplier;
                    }
                }
            }
        }

        //we create the column names
        var sequenceToWords = tokenizer.SequenceToWords(Enumerable.Range(1, embeddingDim)).Select(x => x ?? "OOV")
            .ToList();
        List<string> columns = new();
        for (int i = 1; i <= embeddingDim; ++i)
        {
            var tfidfColumnName = columnNameToEncode + TFIDF_COLUMN_NAME_KEY + (i - 1);
            if (addTokenNameAsColumnNameSuffix)
            {
                tfidfColumnName += "_" + sequenceToWords[i - 1];
            }

            columns.Add(tfidfColumnName);
        }

        var tensor = CpuTensor<float>.New(tfIdf, columns.Count);
        return DataFrame.New(tensor, columns);
    }

    private const string TFIDF_COLUMN_NAME_KEY = "_tfidf_";

}
