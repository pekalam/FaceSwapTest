using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Common;
using FaceRecognitionDotNet;
using NumSharp;
using NumSharp.Utilities;
using OpenCvSharp;
using Point = FaceRecognitionDotNet.Point;


public static class A{
    public static unsafe Mat GetMat(this NDArray narray)
    {
        var dtype = narray.dtype;
        int depth;

        if (dtype == typeof(byte))
        {
            depth = MatType.CV_8U;
        }
        else if (dtype == typeof(sbyte))
        {
            depth = MatType.CV_8S;
        }
        else if (dtype == typeof(short))
        {
            depth = MatType.CV_16S;
        }
        else if (dtype == typeof(ushort))
        {
            depth = MatType.CV_16U;
        }
        else if (dtype == typeof(Int32))
        {
            depth = MatType.CV_32S;
        }
        else if (dtype == typeof(float))
        {
            depth = MatType.CV_32F;
        }
        else if (dtype == typeof(double))
        {
            depth = MatType.CV_64F;
        }
        else
        {
            throw new NotImplementedException($"not support datatype: {dtype}");
        }
        if (narray.shape.Length != 3)
        {
            throw new NotImplementedException($"not support shape.Length: {narray.shape.Length}");
        }
        int row = narray.shape[0];
        int cols = narray.shape[1];
        int channels = narray.shape[2];
        var _mattype = MatType.MakeType(depth, channels);
        return new NaMap(row, cols, _mattype, narray);
    }

    public unsafe class NaMap : Mat
    {
        private readonly NDArray narray;
        public NaMap(int rows, int cols, MatType type, NDArray narray, long step = 0) :
            base(rows, cols, type, (IntPtr)narray.Unsafe.Address, step)
        {
            this.narray = narray;
        }
    }


    [MethodImpl(MethodImplOptions.NoOptimization)]
    private static void DoNothing(Mat ptr)
    {
        var p = ptr;
    }
}



namespace FaceSwapAutoencoder
{
    public record PreprocessedOutput(NDArray face, Rect faceRect, IDictionary<FacePart, IEnumerable<FacePoint>> landmarks, Point2f[] p1, Point2f[] p2);

    public class FaceSwapPreprocessing
    {
        private const int ImgWidth = 96;
        private const int ImgHeight = 96;

        private readonly HeadPoseNormalization _headPoseNormalization = new();
        private Mat _resizedMat;
        private bool _normalize = true;
        private AccurateFaceDetection _faceDetection;

        public FaceSwapPreprocessing(bool normalize = true, Rect? initialFaceLocation = null)
        {
            _faceDetection = new AccurateFaceDetection(attractingRoi: initialFaceLocation);
            _normalize = normalize;
            SharedFaceRecognitionModel.Init(new FaceRecognitionModelSettings());
            _resizedMat = Mat.Zeros(ImgHeight, ImgWidth, MatType.CV_8UC3);
        }

        public Mat InverseAffine(NDArray modelOutput, PreprocessedOutput preprocessedOutput)
        {
            Debug.Assert(preprocessedOutput.p1 != null && preprocessedOutput.p2 != null && preprocessedOutput.faceRect != null);

            var normalized = modelOutput.GetMat();

            normalized.ConvertTo(normalized, MatType.CV_8UC3);

            var p1 = preprocessedOutput.p1.Clone() as Point2f[];
            var p2 = preprocessedOutput.p2.Clone() as Point2f[];

            for (int i = 0; i < preprocessedOutput.p1.Length; i++)
            {
                p1[i] = new Point2f((preprocessedOutput.p1[i].X - preprocessedOutput.faceRect.X) * (96.0f / preprocessedOutput.faceRect.Width),
                    (preprocessedOutput.p1[i].Y - preprocessedOutput.faceRect.Y) * (96.0f / preprocessedOutput.faceRect.Height));
                p2[i] = new Point2f(preprocessedOutput.p2[i].X * (96.0f / preprocessedOutput.faceRect.Width),
                    preprocessedOutput.p2[i].Y * (96.0f / preprocessedOutput.faceRect.Height));
            }

            var h = Cv2.GetAffineTransform(p2, p1);

            // Cv2.CvtColor(normalized, normalized, ColorConversionCodes.BGR2BGRA);
            // var mat = new Mat(new []{96,96}, MatType.CV_8UC4, new Scalar(0,0,0,0));
            var mat = new Mat(new []{96,96}, MatType.CV_8UC3, new Scalar(0,0,0));
            Cv2.WarpAffine(normalized, mat, h, normalized.Size(), InterpolationFlags.Linear,
                BorderTypes.Transparent, borderValue: new Scalar(0,0,0));

            //Cv2.ImShow("w", mat);
            // Cv2.WaitKey();

            return mat;
        }

        public PreprocessedOutput? Preprocess(Mat photo, bool masked = false)
        {
            var faceDetectionOut = _faceDetection.DetectFace(photo);

            if (faceDetectionOut == null)
            {
                return null;
            }

            if (masked)
            {
                var maskedImg = MaskUtils.GetMasked(photo, faceDetectionOut.landmarks);
                if (maskedImg == null)
                {
                    return null;
                }
                photo = maskedImg;
            }

            Mat normalized;
            Point2f[]? p1 = null, p2 = null;
            if (_normalize)
            {
                (normalized,p1,p2) = _headPoseNormalization.NormalizePosition(photo, faceDetectionOut.faceRect, faceDetectionOut.landmarks);
            }
            else
            {
                normalized = new Mat(photo, faceDetectionOut.faceRect);
            }

            Cv2.Resize(normalized, _resizedMat, new Size(ImgWidth, ImgHeight));
            Cv2.CvtColor(_resizedMat, _resizedMat, ColorConversionCodes.BGRA2RGB);
            var buffer = new byte[_resizedMat.Rows * _resizedMat.Cols * _resizedMat.Channels()];
            Marshal.Copy(_resizedMat.Data, buffer, 0, buffer.Length);

            //Cv2.ImShow("x", _resizedMat);
            //Cv2.WaitKey();

            var npNormalized = np.array(buffer, np.uint8).astype(np.float32).reshape(96,96,3);
            return new(npNormalized, faceDetectionOut.faceRect, faceDetectionOut.landmarks, p1,p2);
        }
    }


   

}