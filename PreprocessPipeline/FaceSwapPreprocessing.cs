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
    public record PreprocessedOutput(NDArray? face, Rect? faceRect, List<IDictionary<FacePart, IEnumerable<FacePoint>>>? landmarks, Point2f[]? p1, Point2f[]? p2);

    public class FaceSwapPreprocessing
    {
        private const int ImgWidth = 96;
        private const int ImgHeight = 96;

        private readonly HeadPoseNormalization _headPoseNormalization = new();
        private readonly HcFaceDetection _faceDetection = new(new HcFaceDetectionSettings());
        private ArrayPool<byte> _memoryPool;
        private Mat _resizedMat;
        private bool _normalize = true;

        public FaceSwapPreprocessing(bool normalize = true)
        {
            _normalize = normalize;
            _memoryPool = ArrayPool<byte>.Shared;
            SharedFaceRecognitionModel.Init(new FaceRecognitionModelSettings());
            _resizedMat = Mat.Zeros(ImgHeight, ImgWidth, MatType.CV_8UC3);
        }

        private Image LoadImage(Mat photo)
        {
            var buffer = _memoryPool.Rent(photo.Rows * photo.Cols * photo.ElemSize());
            Marshal.Copy(photo.Data, buffer, 0, buffer.Length);
            var img = FaceRecognition.LoadImage(buffer, photo.Rows, photo.Cols, photo.ElemSize(), Mode.Rgb);
            return img;
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
                p1[i] = new Point2f((preprocessedOutput.p1[i].X - preprocessedOutput.faceRect.Value.X) * (96.0f / preprocessedOutput.faceRect.Value.Width),
                    (preprocessedOutput.p1[i].Y - preprocessedOutput.faceRect.Value.Y) * (96.0f / preprocessedOutput.faceRect.Value.Height));
                p2[i] = new Point2f(preprocessedOutput.p2[i].X * (96.0f / preprocessedOutput.faceRect.Value.Width),
                    preprocessedOutput.p2[i].Y * (96.0f / preprocessedOutput.faceRect.Value.Height));
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

        private Rect SelectRect(Rect[] rects, List<IDictionary<FacePart, IEnumerable<FacePoint>>> landmarks)
        {
            var rect = rects.OrderBy(v1 => v1.Width * v1.Height).Last();
            return rect;
        }
        
        private Rect SelectRect2(Rect[] rects, List<IDictionary<FacePart, IEnumerable<FacePoint>>> landmarks)
        {
            var consideredFaceParts = new[]
                {FacePart.LeftEye, FacePart.RightEye, FacePart.TopLip, FacePart.LeftEyebrow, FacePart.RightEyebrow};
            var selectedRects = new List<Rect>();

            foreach (var landmark in landmarks.Where(l => l.Keys.Intersect(consideredFaceParts).All(consideredFaceParts.Contains)))
            {
                foreach (var rect in rects)
                {
                    var contains = true;
                    foreach (var (k, v) in landmark.Where((kv) => consideredFaceParts.Contains(kv.Key)))
                    {
                        if (!v.All(p => rect.Contains(p.Point.X, p.Point.Y)))
                        {
                            contains = false;
                            break;
                        }
                    }

                    if (contains && !selectedRects.Contains(rect))
                    {
                        selectedRects.Add(rect);
                    }
                }
            }

            selectedRects.Sort((v1, v2) => v1.Width * v1.Height <= v2.Width * v2.Height ? -1 : 1);
            if (selectedRects.Count == 0)
            {
                return Rect.Empty;
            }
            return selectedRects[^1];
        }

        public PreprocessedOutput Preprocess(Mat photo)
        {
            var rects = _faceDetection.DetectFrontalThenProfileFaces(photo);
            if (rects.Length == 0)
            {
                return new(null, null, null, null,null);
            }

            var landmarks = SharedFaceRecognitionModel.Model
                .FaceLandmark(LoadImage(photo)).ToList();
            if (landmarks.Count == 0)
            {
                return new(null, null, null, null, null);
            }

            var rect = SelectRect2(rects, landmarks);
            if (rect == Rect.Empty)
            {
                return new(null, null, null, null, null);
            }

            Mat normalized;
            Point2f[]? p1 = null, p2 = null;
            if (_normalize)
            {
                (normalized,p1,p2) = _headPoseNormalization.NormalizePosition(photo, rect, landmarks);
            }
            else
            {
                normalized = new Mat(photo, rect);
            }

            Cv2.Resize(normalized, _resizedMat, new Size(ImgWidth, ImgHeight));
            Cv2.CvtColor(_resizedMat, _resizedMat, ColorConversionCodes.BGRA2RGB);
            var buffer = new byte[_resizedMat.Rows * _resizedMat.Cols * _resizedMat.Channels()];
            Marshal.Copy(_resizedMat.Data, buffer, 0, buffer.Length);

            //Cv2.ImShow("x", _resizedMat);
            //Cv2.WaitKey();

            var npNormalized = np.array(buffer, np.uint8).astype(np.float32).reshape(96,96,3);
            return new(npNormalized, rect, landmarks, p1,p2);
        }
    }


   

}