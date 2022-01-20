using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace SharpNet.HPO
{
    public class GridSearchHPO<T> : AbstractHpo<T> where T : class
    {
        #region private fields
        private int _nextSampleIndex;
        #endregion

        public GridSearchHPO(IDictionary<string, object> searchSpace, Func<T> createDefaultSample, Action<T> postBuild, Func<T, bool> isValidSample) 
            : base(searchSpace, createDefaultSample, postBuild, isValidSample)
        {
        }

        protected override T Next
        {
            get
            {
                if (_nextSampleIndex >= SearchSpaceSize)
                {
                    return null;
                }
                var currentSampleIndex = _nextSampleIndex++;
                Debug.Assert(currentSampleIndex >= 0);
                Debug.Assert(currentSampleIndex < SearchSpaceSize);
                var sample = new Dictionary<string, string>();
                foreach (var (parameterName, parameterValues) in SearchSpace.OrderBy(e => e.Key))
                {
                    sample[parameterName] = parameterValues.HyperParameterStringValueAtIndex(currentSampleIndex % parameterValues.Length);
                    _nextSampleIndex /= parameterValues.Length;
                }
                var t = CreateDefaultSample();
                ClassFieldSetter.Set(t, FromString2String_to_String2Object(sample));
                PostBuild(t);
                return IsValidSample(t) ? t : Next;
            }
        }

        private int SearchSpaceSize
        {
            get
            {
                int result = 1;
                foreach (var v in SearchSpace.Values)
                {
                    result *= v.Length;
                }
                return result;
            }
        }
    }
}