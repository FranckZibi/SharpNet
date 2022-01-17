using System;
using System.Collections.Generic;
using System.Linq;

namespace SharpNet.HPO
{
    public class RandomGridSearchHPO<T> : AbstractHPO<T> where T : class, new()
    {
        #region private fields
        private readonly Random _rand = new();
        private readonly HashSet<string> _processedSpaces = new();
        #endregion

        public RandomGridSearchHPO(IDictionary<string, object> searchSpace, Func<T> createDefaultHyperParameters, Func<T, bool> isValid) : 
            base(searchSpace, createDefaultHyperParameters, isValid)
        {
        }

        protected override T Next
        {
            get
            {
                //we'll make '1000' tries to retrieve a new and valid hyper parameter space
                for (int i = 0; i < 1000; ++i)
                {
                    var searchSpaceHyperParameters = new Dictionary<string, string>();
                    foreach (var (parameterName, parameterSearchSpace) in _searchSpace.OrderBy(l => l.Key))
                    {
                        searchSpaceHyperParameters[parameterName] = parameterSearchSpace.GetRandomSearchSpaceHyperParameterStringValue(_rand);
                    }
                    //we ensure that we have not already processed this search space
                    var searchSpaceHash = ComputeHash(searchSpaceHyperParameters);
                    lock (_processedSpaces)
                    {
                        if (!_processedSpaces.Add(searchSpaceHash))
                        {
                            continue; //already processed before
                        }
                    }
                    var t = _createDefaultHyperParameters();
                    ClassFieldSetter.Set(t, FromString2String_to_String2Object(searchSpaceHyperParameters));
                    if (_isValid(t))
                    {
                        return t;
                    }
                }
                return null;
            }
        }
    }
}
