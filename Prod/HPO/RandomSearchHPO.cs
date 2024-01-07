using System;
using System.Collections.Generic;
using System.Linq;
using SharpNet.Hyperparameters;

namespace SharpNet.HPO
{
    public class RandomSearchHPO : AbstractHpo
    {
        #region private fields
        private readonly HyperparameterSearchSpace.RANDOM_SEARCH_OPTION _randomSearchOption;
        private readonly Random _rand = new();
        private readonly HashSet<string> _processedSpaces = new();
        #endregion

        public RandomSearchHPO(IDictionary<string, object> searchSpace,
            Func<ISample> createDefaultSample,
            string workingDirectory,
            HyperparameterSearchSpace.RANDOM_SEARCH_OPTION randomSearchOption = HyperparameterSearchSpace.RANDOM_SEARCH_OPTION.PREFER_MORE_PROMISING) : 
            base(searchSpace, createDefaultSample, workingDirectory)
        {
            _randomSearchOption = randomSearchOption;
        }

        protected override (ISample,int, string) Next
        {
            get
            {
                //we'll make '1000' tries to retrieve a new and valid hyper parameter space
                for (int i = 0; i < 1000; ++i)
                { 
                    var searchSpaceHyperparameters = new Dictionary<string, string>();
                    foreach (var (parameterName, parameterSearchSpace) in SearchSpace.OrderBy(l => l.Key))
                    {
                        searchSpaceHyperparameters[parameterName] = parameterSearchSpace.Next_SampleStringValue(_rand, _randomSearchOption);
                    }
                    var sample = CreateDefaultSample();
                    sample.Set(Utils.FromString2String_to_String2Object(searchSpaceHyperparameters));

                    //we try to fix inconsistencies in the sample
                    if (!sample.FixErrors())
                    {
                        continue; //we failed to fix the inconsistencies in the sample : we have to discard it 
                    }

                    //we ensure that we have not already processed this search space
                    lock (_processedSpaces)
                    {
                        if (!_processedSpaces.Add(sample.ComputeHash()))
                        {
                            continue; //already processed before
                        }
                    }

                    var sampleDescription = ToSampleDescription(searchSpaceHyperparameters, sample);
                    return (sample, _nextSampleId++, sampleDescription);
                }
                return (null,-1, "");
            }
        }
    }
}
