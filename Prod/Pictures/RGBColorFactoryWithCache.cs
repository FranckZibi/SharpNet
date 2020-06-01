namespace SharpNet.Pictures
{
    public class RGBColorFactoryWithCache
    {
        private readonly bool mustBeThreadSafe;
        private readonly RGBColor[][][] cache;

        public RGBColorFactoryWithCache(bool mustBeThreadSafe)
        {
            this.mustBeThreadSafe = mustBeThreadSafe;
            cache = new RGBColor[256][][];
            for (int red = 0; red < 256; ++red)
            {
                cache[red] = new RGBColor[256][];
            }
        }
        public RGBColor Build(byte red, byte green, byte blue)
        {
            var currentBlueInCache = cache[red][green];
            if (currentBlueInCache == null)
            {
                currentBlueInCache = new RGBColor[256];
                if (mustBeThreadSafe)
                {
                    lock (cache[red])
                    {
                        var tmp = cache[red][green];
                        if (tmp == null)
                        {
                            cache[red][green] = currentBlueInCache;
                        }
                        else
                        {
                            currentBlueInCache = tmp;
                        }
                    }
                }
                else
                {
                    cache[red][green] = currentBlueInCache;
                }
            }
            var currentInCache = currentBlueInCache[blue];
            if (currentInCache != null)
            {
                return currentInCache;
            }

            var color = new RGBColor(red, green, blue);
            currentBlueInCache[blue] = color;
            return color;
        }
    }
}