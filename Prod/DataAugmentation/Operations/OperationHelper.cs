using System;
using System.Collections.Generic;
using System.Linq;

namespace SharpNet.DataAugmentation.Operations
{
    public static class OperationHelper
    {
        public static void CheckIntegrity(List<Operation> operations)
        {
            if (operations.Count == 0)
            {
                return; //empty operation list is allowed
            }

            operations = operations.ToList(); //defensive copy
            //we remove all Cutout operations at the end of the list
            while (operations.Count != 0 && operations.Last().GetType() == typeof(Cutout))
            {
                operations.RemoveAt(operations.Count-1);
            }

            var dico = operations.GroupBy(x => x.GetType()).ToDictionary(x => x.Key, x => x.Count());

            //the Cutout operation (when it is used) must be the last one in the least
            if (dico.ContainsKey(typeof(Cutout)))
            {
                throw new ArgumentException(typeof(Cutout) + " operation must be the last one in the list");
            }
            //CutMix and MixUp can't be used at the same time
            if (dico.ContainsKey(typeof(CutMix)) && dico.ContainsKey(typeof(MixUp)))
            {
                throw new ArgumentException(typeof(CutMix) + " and "+typeof(MixUp)+" can't appear at the same time in the list");
            }

            //CutMix and MixUp must be the last operations (only Cutout is allowed to be after them)
            foreach (var t in new[] {typeof(CutMix), typeof(MixUp)})
            {
                if (dico.ContainsKey(t) && operations.Last().GetType() != t)
                {
                    throw new ArgumentException(t + " operation must be the last operation");
                }
            }
        }
    }
}