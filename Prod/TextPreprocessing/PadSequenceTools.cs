using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using SharpNet.CPU;

namespace SharpNet.TextPreprocessing
{
    public static class PadSequenceTools
    {
        /// <summary>
        /// Pads sequences to the same length
        /// Using the same API as in Keras (see: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence/pad_sequences)
        /// </summary>
        /// <param name="sequences">List of sequences (each sequence is a list of 'T').</param>
        /// <param name="maxLen">Optional Int, maximum length of all sequences.
        /// If not provided (less or equal then 0), sequences will be padded to the length of the longest individual sequence.</param>
        /// <param name="isPrePadding">if the sequence is too short, should we use pre padding or post padding
        /// (optional, defaults to pre padding):
        /// pad either before or after each sequence.
        /// </param>
        /// <param name="isPreTruncating">if the sequence is too long, should we use 'pre' truncating or 'post' truncating
        /// (optional, defaults to 'pre')
        /// remove values from sequences larger than maxLen, either at the beginning or at the end of the sequences.
        /// </param>
        /// <returns></returns>
        public static CpuTensor<T> PadSequence<T>(List<List<T>> sequences, int maxLen = -1, bool isPrePadding = true, bool isPreTruncating = true)
        {
            if (maxLen <= 0)
            {
                maxLen = sequences.Select(s => s.Count).Max();
            }
            var result = new CpuTensor<T>(new []{sequences.Count, maxLen});
            void ProcessRow(int row)
            {
                var sequence = sequences[row];
                int firstRowFromSequence;
                int lastRowFromSequence;
                int firstRowInResultSpan;
                if (sequence.Count <= maxLen)
                {
                    //the sequence is too short: we need to pad it
                    firstRowFromSequence = 0;
                    lastRowFromSequence = sequence.Count - 1;
                    firstRowInResultSpan = isPrePadding ? (maxLen - sequence.Count) : 0;
                }
                else
                {
                    //the sequence is too long: we need to truncate it
                    firstRowFromSequence = isPreTruncating ? (sequence.Count-maxLen) : 0;
                    lastRowFromSequence = firstRowFromSequence+maxLen-1;
                    firstRowInResultSpan = 0;
                }

                int indexInResultSpan = result.Idx(row, firstRowInResultSpan);
                var resultSpan = result.SpanContent;
                for (int i = firstRowFromSequence; i <= lastRowFromSequence; ++i)
                {
                    resultSpan[indexInResultSpan++] = sequence[i];
                }
            }
            Parallel.For(0, sequences.Count, ProcessRow);
            return result;
        }
    }
}