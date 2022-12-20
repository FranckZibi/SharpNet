using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
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


        private readonly HashSet<string> _stopWords = new()
        {
            "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
            "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they",
            "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those",
            "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
            "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at",
            "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above",
            "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then",
            "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most",
            "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
            "can", "will", "just", "don", "should", "now"
        };

        private readonly IStemmer _stemmer;
        private readonly ConcurrentDictionary<string, string> _wordToStem = new();

        /// <summary>
        /// characters that will be filtered (removed) from input
        /// </summary>
        private readonly char[] _filters;
        private readonly ConcurrentDictionary<string, int> _wordToWordCount =  new ConcurrentDictionary<string, int>();
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
            string filters = "  !\"#$%&()*+,-—.…/:;<=>?@[\\]^_{|}~\t\n\r´`‘’'",
            IStemmer stemmer = null)
        {
            _numWords = numWords;
            _oovToken = oovToken;
            _lowerCase = lowerCase;
            _filters = filters.ToCharArray();
            _stemmer = stemmer;
        }

        public void FitOnTexts(IList<string> texts)
        {
            void FitOnText(string text)
            {
                foreach (var word in ExtractWords(text))
                {
                    _wordToWordCount.AddOrUpdate(word, 1, (_, oldValue) => oldValue + 1);
                }
                _wordToWordIndex = null;
            }
            Parallel.For(0, texts.Count, i => FitOnText(texts[i]));
        }


        public static string RemoveDiacritics(string text)
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
                    if (!_wordToStem.TryGetValue(result[i], out var stemmedWord))
                    {
                        try
                        {
                            stemmedWord = _stemmer.Stem(result[i]).Value;
                        }
                        catch (Exception)
                        {
                            Console.WriteLine($"Fail to do stemming for word: {result[i]}");
                            stemmedWord = result[i];
                        }
                        if (_stopWords.Contains(stemmedWord) || _stopWords.Contains(result[i]))
                        {
                            stemmedWord = "";
                        }
                        _wordToStem[result[i]] = stemmedWord;
                    }

                    result[i] = stemmedWord;
                }
            }
            return result.Where(s=>!string.IsNullOrEmpty(s)).ToArray();
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

       

        public List<List<int>> TextsToSequences(IList<string> texts)
        {
            var _ = WordIndex;
            var res = new List<int>[texts.Count].ToList();
            Parallel.For(0, texts.Count, i => res[i] = TextToSequence(texts[i]));
            return res;
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
