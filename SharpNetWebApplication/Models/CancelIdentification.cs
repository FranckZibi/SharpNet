using System;

namespace SharpNetWebApplication.Models
{
    public class CancelIdentification
    {
        public string Id { get; set; }
        public bool IsDone { get; set; }
        public string Prediction { get; set; }
        public double PredictionProbability { get; set; }
        public DateTime StartComputationDate { get; set; }
        public int ComputationTimeInMilliseconds { get; set; } = 0;
    }
}
