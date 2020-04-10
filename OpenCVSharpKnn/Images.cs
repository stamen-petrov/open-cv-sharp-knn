using OpenCvSharp;
using System;
using System.Collections.Generic;

namespace OpenCVSharpKnn
{
    public class Images : IDisposable
    {
        public List<Mat> Train { get; }

        public List<Mat> Test { get; }

        private Images()
        {
            Train = new List<Mat>();
            Test = new List<Mat>();
        }

        public static Images GetImages(string inputFile, 
            int hImagesCount, int vImagesCount,
            int imageWidth, int imageHeight)
        {
            var result = new Images();
            using var img = Cv2.ImRead(inputFile);
            Cv2.CvtColor(img, img, ColorConversionCodes.BGR2GRAY);

            result.SetImageLists(img,
                hImagesCount, vImagesCount,
                imageWidth, imageHeight);            

            return result;
        }

        private void SetImageLists(Mat img,
            int hImagesCount, int vImagesCount,
            int width, int height)
        {
            for (int i = 0; i < hImagesCount; i++)
            {
                for (int j = 0; j < vImagesCount; j++)
                {
                    var image = img[
                        new Rect(i * width, j * height, width, height)
                    ];

                    if (i < hImagesCount / 2)
                    {
                        Train.Add(image);
                    }
                    else
                    {
                        Test.Add(image);
                    }                    
                }
            }
        }


        #region IDisposable

        private bool _disposed;

        ~Images()
        {
            Dispose(false);
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (disposing && !_disposed)
            {
                if (Test != null)
                {
                    foreach(var mat in Test)
                    {
                        mat?.Dispose();
                    }                    
                }

                if (Train != null)
                {
                    foreach(var mat in Train)
                    {
                        mat?.Dispose();
                    }
                }

                _disposed = true;
            }
        }

        #endregion
    }
}
