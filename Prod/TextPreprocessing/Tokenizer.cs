using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using Porter2StemmerStandard;

namespace SharpNet.TextPreprocessing
{
    /// <summary>
    /// Text tokenization utility class.
    /// Using the same API as in Keras (see: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer)
    /// </summary>
    public class Tokenizer
    {
        #region private fields
        /// <summary>
        /// The maximum number of words to keep, based on word frequency.
        /// Only the most common '_numWords-1' words will be kept.
        /// </summary>
        private readonly int _numWords;
        /// <summary>
        /// if null or empty (default value):
        ///     out-of-vocabulary words will simply be discarded (removed) from input
        ///     Example:
        ///         "This *!dfztà idea" => "This idea"
        /// else:
        ///     out-of-vocabulary words will be replaced by this '_oovToken' string
        ///     (and this '_oovToken' string will be added to WordIndex dictionary with index == 1 )
        ///     Example:
        ///         "This *!dfztà idea" => "This "+_oovToken+" idea"
        /// </summary>
        private readonly string _oovToken;

        /// <summary>
        /// whether to convert the texts to lowercase
        /// default: true
        /// </summary>
        private readonly bool _lowerCase;

        private readonly IStemmer _stemmer;

        /// <summary>
        /// characters that will be filtered (removed) from input
        /// </summary>
        private readonly char[] _filters;
        private readonly IDictionary<string, int> _wordToWordCount =  new Dictionary<string, int>();
        /// <summary>
        /// index associated with each word, from the most common (index 1 or 2) to the least common (index _numWords-1)
        /// if no out-of-vocabulary is used (_oovToken == null)
        ///     index of padding token:                 0
        ///     index of most common word:              1
        ///     index of 2nd most common word:          2
        ///     index of least common word:             _numWords-1
        /// else
        ///     index of padding token:                 0
        ///     index of out-of-vocabulary token:       1
        ///     index of most common word:              2
        ///     index of 2nd most common word:          3
        ///     index of least common word:             _numWords-1
        /// </summary>
        private IDictionary<string, int> _wordToWordIndex =  new Dictionary<string, int>();
        #endregion


        public int DistinctWords => _wordToWordIndex.Count;

        public Tokenizer(
            int numWords = int.MaxValue,
            string oovToken = null, 
            bool lowerCase = true, 
            string filters = " !\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r’",
            IStemmer stemmer = null)
        {
            _numWords = numWords;
            _oovToken = oovToken;
            _lowerCase = lowerCase;
            _filters = filters.ToCharArray();
            _stemmer = stemmer;
        }

        public void FitOnTexts(IEnumerable<string> texts)
        {
            foreach (var text in texts)
            {
                FitOnText(text);
            }
        }


        private static string RemoveDiacritics(string text)
        {
            var normalizedString = text.Normalize(NormalizationForm.FormD);
            var stringBuilder = new StringBuilder(capacity: normalizedString.Length);

            for (int i = 0; i < normalizedString.Length; i++)
            {
                char c = normalizedString[i];
                var unicodeCategory = CharUnicodeInfo.GetUnicodeCategory(c);
                if (unicodeCategory != UnicodeCategory.NonSpacingMark)
                {
                    stringBuilder.Append(c);
                }
            }

            return stringBuilder
                .ToString()
                .Normalize(NormalizationForm.FormC);
        }

        // public for testing only
        public string[] ExtractWords(string text)
        {
            //we remove accents
            text = RemoveDiacritics(text);

            if (_lowerCase)
            {
                text = text.ToLowerInvariant();
            }
            var result = text.Split(_filters, StringSplitOptions.RemoveEmptyEntries);
            if (_stemmer != null)
            {
                for (int i = 0; i < result.Length; i++)
                {
                    if (result[i].All(char.IsLetterOrDigit))
                    {
                        result[i] = _stemmer.Stem(result[i]).Value;
                    }
                }
            }
            return result;
        }

        //public List<List<int>> FitOnTextsAndTextsToSequences(IEnumerable<string> texts)
        //{
        //    List<string> _allWord = new();
        //    List<int> _allWordCount = new();
        //    Dictionary<string, int> _wordToIndexInAllWordList = new();
        //    List<List<int>> _fitOnTexts = new();
        //    foreach (var text in texts)
        //    {
        //        List<int> res = new();
        //        foreach (var word in ExtractWords(text))
        //        {
        //            if (_wordToIndexInAllWordList.TryGetValue(word, out var indexInAllWordList))
        //            {
        //                res.Add(indexInAllWordList);
        //                ++_allWordCount[indexInAllWordList];
        //            }
        //            else
        //            {
        //                _allWord.Add(word);
        //                _allWordCount.Add(1);
        //                indexInAllWordList = res.Count;
        //                res.Add(indexInAllWordList);
        //                _wordToIndexInAllWordList[word] = indexInAllWordList;
        //            }
        //        }
        //        _fitOnTexts.Add(res);
        //    }

        //    var oldIdxAndCount = new List<Tuple<int, int>>();
        //    for (int i = 0; i < _allWordCount.Count; ++i)
        //    {
        //        oldIdxAndCount.Add(new Tuple<int, int>(i, _allWordCount[i]));
        //    }



        //    int oovOldIndex = -1;
        //    int newWordIndex = 1;
        //    var oldIdxToWordIndex = new Dictionary<int, int>();
        //    if (!string.IsNullOrEmpty(_oovToken))
        //    {
        //        newWordIndex++; // the word index of oov token will always be 1
        //    }
        //    foreach (var (oldIdx, _) in oldIdxAndCount.OrderByDescending(x => x.Item2))
        //    {
        //        if (oldIdx != oovOldIndex)
        //        {
        //            oldIdxToWordIndex[oldIdx] = newWordIndex++;
        //        }
        //    }

            
        //    var finalResults = new List<List<int>>();
        //    foreach (var doc in _fitOnTexts)
        //    {
        //        var result = new List<int>();
        //        foreach (var oldIdx in doc)
        //        {
        //            var wordIndex = oldIdxToWordIndex[oldIdx];
        //            if (wordIndex <= _numWords)
        //            {
        //                result.Add(wordIndex);
        //            }
        //            else
        //            {
        //                if (!string.IsNullOrEmpty(_oovToken))
        //                {
        //                    result.Add(1);
        //                }
        //            }
        //        }
        //        finalResults.Add(result);
        //    }

        //    return finalResults;
        //}

        private void FitOnText(string text)
        {

            //var utf8String = Encoding.UTF8.GetBytes(text);

            ////convert them into unicode bytes.
            //byte[] unicodeBytes = Encoding.Convert(Encoding.UTF8, Encoding.Unicode, utf8String);

            ////builds the converted string.
            //var strText8 = Encoding.Unicode.GetString(unicodeBytes);

            foreach (var word in ExtractWords(text))
            {
                if (_wordToWordCount.ContainsKey(word))
                {
                    ++_wordToWordCount[word];
                }
                else
                {
                    _wordToWordCount[word] = 1;
                }
            }
            _wordToWordIndex = null;
        }

        public List<List<int>> TextsToSequences(IEnumerable<string> texts)
        {
            return texts.Select(TextToSequence).ToList();
        }

        public List<int> TextToSequence(string text)
        {
            var wordToWordIndex = WordIndex;
            var result = new List<int>();
            foreach (var word in ExtractWords(text))
            {
                if (wordToWordIndex.TryGetValue(word, out var wordIndex) && wordIndex <= _numWords)
                {
                    result.Add(wordIndex);
                }
                else
                {
                    if (!string.IsNullOrEmpty(_oovToken))
                    {
                        result.Add(wordToWordIndex[_oovToken]);
                    }
                }
            }
            return result;
        }

        public string SequenceToText(IEnumerable<int> sequence) => string.Join(" ", SequenceToWords(sequence));

        public List<string> SequenceToWords(IEnumerable<int> sequence)
        {
            var wordIndexToToken = new Dictionary<int, string>();
            foreach (var e in WordIndex)
            {
                wordIndexToToken[e.Value] = e.Key;
            }
            return sequence.Select(s => wordIndexToToken.ContainsKey(s) ? wordIndexToToken[s] : _oovToken).ToList();
        }

        public IDictionary<string, int> WordIndex
        {
            get
            {
                if (_wordToWordIndex == null)
                {
                    int newWordIndex = 1;
                    _wordToWordIndex = new Dictionary<string, int>();
                    if (!string.IsNullOrEmpty(_oovToken))
                    {
                        _wordToWordIndex[_oovToken] = newWordIndex++; // the word index of oov token will always be 1
                    }
                    foreach (var e in _wordToWordCount.OrderByDescending(e => e.Value))
                    {
                        _wordToWordIndex[e.Key] = newWordIndex++;
                    }
                }
                return _wordToWordIndex;
            }
        }
    }
}
