using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace SharpNet.HPO
{
    public class GridSearchHPO<T> : AbstractHPO<T> where T : class
    {
        #region private fields
        private int _nextSearchSpaceIndex;
        #endregion

        public GridSearchHPO(IDictionary<string, object> searchSpace, Func<T> createDefaultHyperParameters, Func<T, bool> isValid) 
            : base(searchSpace, createDefaultHyperParameters, isValid)
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
                var currentSearchSpaceIndex = _nextSearchSpaceIndex++;
                Debug.Assert(currentSearchSpaceIndex >= 0);
                Debug.Assert(currentSearchSpaceIndex < SearchSpaceSize);
                var searchSpaceHyperParameters = new Dictionary<string, string>();
                foreach (var (parameterName, parameterValues) in _searchSpace.OrderBy(e => e.Key))
                {
                    searchSpaceHyperParameters[parameterName] = parameterValues.HyperParameterStringValueAtIndex(currentSearchSpaceIndex % parameterValues.Length);
                    _nextSearchSpaceIndex /= parameterValues.Length;
                }
                var t = _createDefaultHyperParameters();
                ClassFieldSetter.Set(t, FromString2String_to_String2Object(searchSpaceHyperParameters));
                return _isValid(t) ? t : Next;
            }
        }

        private int SearchSpaceSize
        {
            get
            {
                int result = 1;
                foreach (var v in _searchSpace.Values)
                {
                    result *= v.Length;
                }
                return result;
            }
        }
    }
}