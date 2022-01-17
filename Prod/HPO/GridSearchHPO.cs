using System;
using System.Collections.Generic;

namespace SharpNet.HPO
{
    public class GridSearchHPO<T> : AbstractHPO<T> where T : class
    {
        private int _nextSearchSpaceIndex;

        public GridSearchHPO(IDictionary<string, object> searchSpace, Func<T> createDefaultHyperParameters) : base(searchSpace, createDefaultHyperParameters)
        {
        }

        protected override T Next
        {
            get
            {
                if (_nextSearchSpaceIndex >= SearchSpaceSize)
                {
                    return null;
                }
                return GetHyperParameters(_nextSearchSpaceIndex++);
            }
        }
    }
}