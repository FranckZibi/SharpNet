using System;
using System.Collections.Generic;
using System.Linq;

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

        public Tokenizer(
            int numWords = int.MaxValue,
            string oovToken = null, 
            bool lowerCase = true, 
            string filters = " !\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n")
        {
            _numWords = numWords;
            _oovToken = oovToken;
            _lowerCase = lowerCase;
            _filters = filters.ToCharArray();
        }

        public void FitOnTexts(IEnumerable<string> texts)
        {
            foreach (var text in texts)
            {
                FitOnText(text);
            }
        }

        // public for testing only
        public string[] ExtractWords(string text)
        {
            if (_lowerCase)
            {
                text = text.ToLowerInvariant();
            }
            return text.Split(_filters, StringSplitOptions.RemoveEmptyEntries);
        }

        private void FitOnText(string text)
        {
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

        public string SequenceToText(List<int> sequence)
        {
            var wordIndexToToken = new Dictionary<int, string>();
            foreach (var e in WordIndex)
            {
                wordIndexToToken[e.Value] = e.Key;
            }
            return string.Join(" ", sequence.Select(s => wordIndexToToken.ContainsKey(s) ? wordIndexToToken[s] : _oovToken));
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
