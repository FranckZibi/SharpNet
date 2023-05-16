namespace SharpNet.Datasets;

public interface TimeSeriesSinglePoint
{
    string UniqueId { get; }
    string TimeSeriesFamily { get; }
    float TimeSeriesTimeStamp { get; }
    float ExpectedTarget { get; } //y
}