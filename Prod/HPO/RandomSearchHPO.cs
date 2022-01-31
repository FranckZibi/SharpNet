using System;
using System.Collections.Generic;
using System.Linq;

namespace SharpNet.HPO
{
    public class RandomSearchHPO<T> : AbstractHpo<T> where T : class, new()
    {
        #region private fields
        private readonly AbstractHyperParameterSearchSpace.RANDOM_SEARCH_OPTION _randomSearchOption;
        private readonly Random _rand = new();
        private readonly HashSet<string> _processedSpaces = new();
        #endregion

        public RandomSearchHPO(IDictionary<string, object> searchSpace,
            Func<T> createDefaultSample,
            Func<T, bool> postBuild,
            double maxAllowedSecondsForAllComputation,
            string workingDirectory,
            AbstractHyperParameterSearchSpace.RANDOM_SEARCH_OPTION randomSearchOption) : 
            base(searchSpace, createDefaultSample, postBuild, maxAllowedSecondsForAllComputation, workingDirectory, new HashSet<string>())
        {
            _randomSearchOption = randomSearchOption;
        }

        protected override (T,int, string) Next
        {
            get
            {
                //we'll make '1000' tries to retrieve a new and valid hyper parameter space
                for (int i = 0; i < 1000; ++i)
                { 
                    var searchSpaceHyperParameters = new Dictionary<string, string>();
                    foreach (var (parameterName, parameterSearchSpace) in SearchSpace.OrderBy(l => l.Key))
                    {
                        searchSpaceHyperParameters[parameterName] = parameterSearchSpace.Next_SampleStringValue(_rand, _randomSearchOption);
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
                    var sample = CreateDefaultSample();
                    ClassFieldSetter.Set(sample, FromString2String_to_String2Object(searchSpaceHyperParameters));
                    if (PostBuild(sample))
                    {
                        var sampleDescription = ToSampleDescription(searchSpaceHyperParameters, sample);
                        return (sample, _nextSampleId++, sampleDescription);
                    }
                }
                return (null,-1, "");
            }
        }
    }
}
