﻿using System;
using System.Collections.Generic;
using System.Linq;
using SharpNet.HyperParameters;

namespace SharpNet.HPO
{
    public class RandomSearchHPO : AbstractHpo
    {
        #region private fields
        private readonly AbstractHyperParameterSearchSpace.RANDOM_SEARCH_OPTION _randomSearchOption;
        private readonly Random _rand = new();
        private readonly HashSet<string> _processedSpaces = new();
        #endregion

        public RandomSearchHPO(IDictionary<string, object> searchSpace,
            Func<ISample> createDefaultSample,
            string workingDirectory,
            AbstractHyperParameterSearchSpace.RANDOM_SEARCH_OPTION randomSearchOption = AbstractHyperParameterSearchSpace.RANDOM_SEARCH_OPTION.PREFER_MORE_PROMISING) : 
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
                    var searchSpaceHyperParameters = new Dictionary<string, string>();
                    foreach (var (parameterName, parameterSearchSpace) in SearchSpace.OrderBy(l => l.Key))
                    {
                        searchSpaceHyperParameters[parameterName] = parameterSearchSpace.Next_SampleStringValue(_rand, _randomSearchOption);
                    }
                    var sample = CreateDefaultSample();
                    sample.Set(Utils.FromString2String_to_String2Object(searchSpaceHyperParameters));
                    //we ensure that we have not already processed this search space
                    lock (_processedSpaces)
                    {
                        if (!_processedSpaces.Add(sample.ComputeHash()))
                        {
                            continue; //already processed before
                        }
                    }
                    if (sample.PostBuild())
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
