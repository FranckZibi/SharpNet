using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using SharpNet.Datasets;

namespace SharpNet.TextPreprocessing;

public static class TfIdf
{
    //private static readonly IStemmer _stemmer = new EnglishPorter2Stemmer();

    public static DataFrame ToTfIdf(DataFrame df, string columnToProcess, int embeddingDim, bool addTokenNameAsColumnNameSuffix = false)
    {
        var documents = df[columnToProcess].StringCpuTensor().ReadonlyContent.ToArray();
        return ToTfIdf(documents, embeddingDim, columnToProcess, addTokenNameAsColumnNameSuffix);
    }

    public static DataFrame ToTfIdf(IList<string> documents, int embeddingDim, string columnNamePrefix = "", bool addTokenNameAsColumnNameSuffix = false)
    {
        //var tokenizer = new Tokenizer(oovToken: null, lowerCase: false, stemmer: _stemmer);
        var tokenizer = new Tokenizer(oovToken: null, lowerCase: true);


        tokenizer.FitOnTexts(documents);
        var training_sequences = tokenizer.TextsToSequences(documents);

        var tfIdf = new float[documents.Count * embeddingDim];

        // documentsContainingWord[wordIdx] : number of distinct documents containing the word at index 'wordIdx'
        var documentsContainingWord = new float[embeddingDim];

        //first step;
        //  tfIdf[documentId * embeddingDim + wordIdx] : number of time the word at 'wordIdx' appears in document 'documentId'
        for (int documentId = 0; documentId < documents.Count; ++documentId)
        {
            foreach (var wordIdx_starting_at_1 in training_sequences[documentId])
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

                ++tfIdf[documentId * embeddingDim + wordIdx];
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
                if (tf != 0)
                {
                    // % of document containing (at least one time) the word at 'wordIdx'
                    float df = documentsContainingWord[wordIdx] / documents.Count;
                    float idf = (float)Math.Log(1 / df);
                    tfIdf[documentId * embeddingDim + wordIdx] = tf * idf;
                }
            }

        }

        //we create the column names
        var sequenceToWords = tokenizer.SequenceToWords(Enumerable.Range(1, embeddingDim)).Select(x => x ?? "OOV").ToList();
        List<string> columns = new();
        for (int i = 1; i <= embeddingDim; ++i)
        {
            var columnName = "tfidf_"+(i - 1);
            if (!string.IsNullOrEmpty(columnNamePrefix))
            {
                columnName = columnNamePrefix + "_" + columnName;
            }
            if (addTokenNameAsColumnNameSuffix)
            {
                columnName += "_"+sequenceToWords[i-1];
            }
            columns.Add(columnName);
        }
            
        return DataFrame.New(tfIdf, columns);
    }
}