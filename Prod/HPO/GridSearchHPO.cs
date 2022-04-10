using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using SharpNet.HyperParameters;

namespace SharpNet.HPO
{
    public class GridSearchHPO : AbstractHpo
    {
        #region private fields
        private int _nextSearchSpaceIndex;
        #endregion

        public GridSearchHPO(IDictionary<string, object> searchSpace, 
            Func<ISample> createDefaultSample,
            string workingDirectory) 
            : base(searchSpace, createDefaultSample, workingDirectory)
        {
        }

        protected override (ISample,int, string) Next
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
                    t.Set(Utils.FromString2String_to_String2Object(sample));
                    if (t.FixErrors())
                    {
                        var sampleDescription = ToSampleDescription(sample, t);
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