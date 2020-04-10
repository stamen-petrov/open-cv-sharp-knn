using OpenCvSharp;
using OpenCvSharp.ML;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace OpenCVSharpKnn
{
    public class Knn : IDisposable
    {
        private KNearest _knn;

        public void Train(List<Mat> inputs, List<int> labels)
        {
            if ((inputs?.Count ?? 0) == 0)
            {
                throw new InvalidDataException("No images input.");
            }

            if ((labels?.Count ?? 0) == 0)
            {
                throw new InvalidDataException("No labels input.");
            }

            var trainFeatures = GetFeaturesInput(inputs);

            var trainLabels = GetLabelsInput(labels);

            _knn = KNearest.Create();
            _knn.Train(trainFeatures, SampleTypes.RowSample, trainLabels);
        }

        public List<int> Test(List<Mat> inputs, int k = 5)
        {
            if ((inputs?.Count ?? 0) == 0)
            {
                throw new InvalidDataException("No images input.");
            }

            var testInput = GetFeaturesInput(inputs);

            var testLabels = new List<float>();
            OutputArray testOuputLabels = OutputArray.Create<float>(testLabels);

            _knn.FindNearest(testInput, k, testOuputLabels);

            return GetLabelsOutput(testLabels);
        }
        
        public double GetSuccessRate(List<int> expected, List<int> actual)
        {
            if ((expected?.Count ?? 0) == 0)
            {
                throw new InvalidDataException("No expected input.");
            }

            if ((actual?.Count ?? 0) == 0)
            {
                throw new InvalidDataException("No actual input.");
            }

            if (actual?.Count != expected?.Count)
            {
                throw new InvalidDataException("Actual & expected must be the same size.");
            }

            int count = actual?.Count ?? 0;
            int errors = 0;
            for (int i = 0; i < count; i++)
            {
                if (expected?[i] != actual?[i])
                {
                    errors++;
                }
            }

            var successRate = 1 - (errors / (float)count);
            return successRate;
        }

        public int Predict(Mat mat)
        {
            Cv2.CvtColor(mat, mat, ColorConversionCodes.BGR2GRAY);
            return Test(new List<Mat> { mat }).Single();
        }

        private InputArray GetFeaturesInput(List<Mat> inputs)
        {
            if ((inputs?.Count ?? 0) == 0)
            {
                throw new InvalidDataException("No images input.");
            }

            var size = inputs[0];
            if (inputs.Any(x => x.Size().Height != size.Height ||
                x.Size().Width != size.Width))
            {
                throw new InvalidDataException("Images must be same size.");
            }

            int imageBytes = size.Height * size.Width;

            var featuresArray = new float[inputs.Count, imageBytes];
            for (int i = 0; i < inputs.Count; i++)
            {
                inputs[i].GetArray<byte>(out byte[] trainD);
                for (int j = 0; j < imageBytes; j++)
                {
                    featuresArray[i, j] = trainD[j];
                }
            }

            return InputArray.Create<float>(featuresArray);
        }

        private InputArray GetLabelsInput(List<int> labelsInput)
        {
            var labelsArray = new float[labelsInput.Count];
            for (int i = 0; i < labelsInput.Count; i++)
            {
                labelsArray[i] = labelsInput[i];
            }

            return InputArray.Create<float>(labelsArray);
        }

        private List<int> GetLabelsOutput(List<float> labelsOutput)
        {
            var labels = new int[labelsOutput.Count];
            for (int i = 0; i < labelsOutput.Count; i++)
            {
                labels[i] = (int)labelsOutput[i];
            }

            return labels.ToList();
        }

        #region IDisposable

        private bool _disposed;

        ~Knn()
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
                if (_knn != null)
                {
                    _knn.Dispose();
                }

                _disposed = true;
            }
        }

        #endregion
    }
}
