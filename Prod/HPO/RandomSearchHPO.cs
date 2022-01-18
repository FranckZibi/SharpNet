using System;
using System.Collections.Generic;
using System.Linq;

namespace SharpNet.HPO
{
    public class RandomSearchHPO<T> : AbstractHPO<T> where T : class, new()
    {
        private readonly HyperParameterSearchSpace.RANDOM_SEARCH_OPTION _randomSearchOption;

        #region private fields
        private readonly Random _rand = new();
        private readonly HashSet<string> _processedSpaces = new();
        #endregion

        public RandomSearchHPO(IDictionary<string, object> searchSpace, Func<T> createDefaultHyperParameters, Action<T> postBuild, Func<T, bool> isValid, HyperParameterSearchSpace.RANDOM_SEARCH_OPTION randomSearchOption) : 
            base(searchSpace, createDefaultHyperParameters, postBuild, isValid)
        {
            _randomSearchOption = randomSearchOption;
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
                        searchSpaceHyperParameters[parameterName] = parameterSearchSpace.GetRandomSearchSpaceHyperParameterStringValue(_rand, _randomSearchOption);
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
                    _postBuild(t);
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
