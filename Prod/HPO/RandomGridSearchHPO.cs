using System;
using System.Collections.Generic;

namespace SharpNet.HPO
{


    public class RandomGridSearchHPO<T> : AbstractHPO<T> where T : class, new()
    {
        private readonly HashSet<int> _processedConfigIndex = new HashSet<int>();
        private readonly Random _rand= new Random();

        public RandomGridSearchHPO(IDictionary<string, object> searchSpace): base(searchSpace)
        {
        }

        public override T Next
        {
            get
            {
                if (_processedConfigIndex.Count == SearchSpaceSize)
                {
                    return null;
                }

                for (int i=0;i<10*SearchSpaceSize;++i)
                {
                    int configIndex = _rand.Next(SearchSpaceSize);
                    if (_processedConfigIndex.Add(configIndex))
                    {
                        return GetHyperParameters(configIndex);
                    }

                }
                return null;
            }
        }
    }
}