using System;
using System.Collections.Generic;

namespace SharpNet.HPO
{
    public class RandomGridSearchHPO<T> : AbstractHPO<T> where T : class, new()
    {
        private readonly HashSet<int> _processedSearchSpaceIndex = new HashSet<int>();
        private readonly Random _rand= new Random();

        public RandomGridSearchHPO(IDictionary<string, object> searchSpace, Func<T> createDefaultHyperParameters): base(searchSpace, createDefaultHyperParameters)
        {
        }

        protected override T Next
        {
            get
            {
                if (_processedSearchSpaceIndex.Count == SearchSpaceSize)
                {
                    return null;
                }

                for (int i = 0; i < 10 * SearchSpaceSize; ++i)
                {
                    int searchSpaceIndex = _rand.Next(SearchSpaceSize);
                    if (_processedSearchSpaceIndex.Add(searchSpaceIndex))
                    {
                        return GetHyperParameters(searchSpaceIndex);
                    }
                }
                return null;
            }
        }
    }
}
