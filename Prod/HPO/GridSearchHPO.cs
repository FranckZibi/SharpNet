﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using log4net;

namespace SharpNet.HPO
{
    public class GridSearchHPO<T> : AbstractHpo<T> where T : class
    {
        #region private fields
        private int _nextSearchSpaceIndex;
        #endregion

        public GridSearchHPO(IDictionary<string, object> searchSpace, Func<T> createDefaultSample, Func<T, bool> postBuild, ILog log, int maxSamplesToProcess) 
            : base(searchSpace, createDefaultSample, postBuild, log, maxSamplesToProcess, new HashSet<string>())
        {
        }

        protected override (T,int, string) Next
        {
            get
            {
                for(;;)
                {
                    if (_nextSearchSpaceIndex >= SearchSpaceSize)
                    {
                        return (null,-1, "");
                    }
                    var currentSampleIndex = _nextSearchSpaceIndex++;
                    Debug.Assert(currentSampleIndex >= 0);
                    Debug.Assert(currentSampleIndex < SearchSpaceSize);
                    var sample = new Dictionary<string, string>();
                    foreach (var (parameterName, parameterValues) in SearchSpace.OrderBy(e => e.Key))
                    {
                        sample[parameterName] = parameterValues.SampleStringValue_at_Index_For_GridSearch(currentSampleIndex % parameterValues.LengthForGridSearch);
                        _nextSearchSpaceIndex /= parameterValues.LengthForGridSearch;
                    }
                    var t = CreateDefaultSample();
                    ClassFieldSetter.Set(t, FromString2String_to_String2Object(sample));
                    if (PostBuild(t))
                    {
                        var sampleDescription = ToSampleDescription(sample);
                        return (t, _nextSampleId++, sampleDescription);
                    }
                };
            }
        }

        private int SearchSpaceSize
        {
            get
            {
                int result = 1;
                foreach (var v in SearchSpace.Values)
                {
                    result *= v.LengthForGridSearch;
                }
                return result;
            }
        }
    }
}