using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using FaceRecognitionDotNet;
using NumSharp;
using OpenCvSharp;
using Tensorflow;
using static Tensorflow.Binding;
using Point = OpenCvSharp.Point;


namespace FaceSwapAutoencoder
{
    public class FaceSwapAutoencoder : IDisposable
    {
        private Graph _graph;
        private Session _session;

        public record Output(NDArray p1, NDArray p2, PreprocessedOutput preprocessedOutput);

        public FaceSwapAutoencoder(string graphModelFilePath, Rect? initialFaceLocation = null)
        {
            Preprocessing = new FaceSwapPreprocessing(initialFaceLocation: initialFaceLocation);
            _graph = new Graph().ImportGraphDef(graphModelFilePath);

            _session = tf.Session(_graph);
            // var nd = np.ones(new Shape(1, 96, 96, 3));
            // Console.WriteLine("Model warmup...");
            // CallModel(nd);
            // foreach (var op in _graph.get_operations())
            // {
            //     Console.WriteLine(op.name);
            // }
        }

        public FaceSwapPreprocessing Preprocessing { get; }

        private (NDArray p1, NDArray p2) CallModel(NDArray image)
        {
            var inputOperation = _graph.get_operation_by_name("import/x");
            var outputP1Operation = _graph.get_operation_by_name("import/autoenc_light/sequential_2/conv2d_transpose_3/Sigmoid");
            var outputP2Operation = _graph.get_operation_by_name("import/autoenc_light/sequential_3/conv2d_transpose_7/Sigmoid");

            NDArray result_p1 = null, result_p2 = null;
            (result_p1, result_p2) = _session.run((outputP1Operation.outputs[0], outputP2Operation.outputs[0]),
                new FeedItem(inputOperation.outputs[0], image));

            return (result_p1, result_p2);
        }

        public Output? Call(Mat image)
        {
            var preprocessed = Preprocessing.Preprocess(image);


            if (preprocessed == null)
            {
                return null;
            }

            var (p1, p2) = CallModel(preprocessed.face!);

            //Preprocessing.InverseAffine(np.squeeze(p2*255.0f), preprocessed);

            return new(p1, p2, preprocessed);
        }

        public void Dispose()
        {
            _session.Dispose();
        }

    }
}
