using System.Collections.Generic;

namespace SharpNet.HPO
{
    public class GridSearchHPO<T> : AbstractHPO<T> where T : class, new()
    {
        private int _nextSearchSpaceIndex;

        public GridSearchHPO(IDictionary<string, object> searchSpace) : base(searchSpace)
        {
        }
        public override T Next
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