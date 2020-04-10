using OpenCvSharp;
using System;
using System.Collections.Generic;

namespace OpenCVSharpKnn
{
    /// <summary>
    /// OpenCvSharp version of handwritten recognition using KNN.
    /// https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_ml/py_knn/py_knn_opencv/py_knn_opencv.html
    /// </summary>
    class Program
    {
        /// <summary>
        /// https://github.com/opencv/opencv/blob/master/samples/data/digits.png
        /// </summary>
        static string _inputAllDigitsFile = "../../../data/digits.png";

        /// <summary>
        /// Sample input digit
        /// </summary>
        static string _inputSingleFile = "../../../data/9_sample.png";

        static Knn _knn = new Knn();

        static void Main(string[] args)
        {
            var images = Images.GetImages(_inputAllDigitsFile, 100, 50, 20, 20);

            TrainKnn(images);

            TestMultipleKnn(images);

            TestSingleKnn(new Mat(_inputSingleFile));

        }

        private static void TrainKnn(Images images)
        {
            var expectedLabels = GetExpectedLabels(images);

            _knn.Train(images.Train, expectedLabels);
        }

        private static void TestSingleKnn(Mat mat)
        {
            var actualLabel = _knn.Predict(mat);
            Console.WriteLine($"Actual label: {actualLabel}");
        }

        private static void TestMultipleKnn(Images images)
        {
            var actualLabels = _knn.Test(images.Test);
            var expectedLabels = GetExpectedLabels(images);

            var success = _knn.GetSuccessRate(expectedLabels, actualLabels);

            Console.WriteLine($"Success rate: {success:P}");
        }

        private static List<int> GetExpectedLabels(Images images)
        {
            var expectedLabels = new List<int>();
            for (int i = 0; i < images.Train.Count; i++)
            {
                expectedLabels.Add((i / 5) % 10);
            }
            return expectedLabels;
        }
    }
}
