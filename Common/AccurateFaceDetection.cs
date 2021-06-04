using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using FaceRecognitionDotNet;
using HDF.PInvoke;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using OpenCvSharp;
using Python.Runtime;
using Tensorflow;
using static Common.Utils;
using Point = FaceRecognitionDotNet.Point;

namespace Common
{
    public record FaceCoords(int leftEyeX, int leftEyeY, int leftCheekX, int leftCheekY, int downChinX, int downChinY,
        int rightEyeX, int rightEyeY, int rightCheekX, int rightCheekY);

    public record FaceDetectionOutput(Rect faceRect, FaceCoords coords, IDictionary<FacePart, IEnumerable<FacePoint>> landmarks);

    public class AccurateFaceDetection : IDisposable
    {
        private readonly ILogger _logger;
        private readonly HcFaceDetection _hc;
        private FaceQNet _faceQNet;
        private FaceDetectionOutput? _prevOutput;
        private double _avgArea;
        private double _areaSum;
        private int _totalProcessed;
        private Rect? _attractingRoi;
        private bool bufInitialized;
        private byte[] buff;
        private FaceRecognition _recog;


        public static readonly FacePart[] ConsideredFaceParts =
            {FacePart.LeftEye, FacePart.RightEye, FacePart.Chin};

        public const double FaceqAcceptedScore = 0.20f;

        private class CandidateRectLandmarks
        {
            public int IntersectLandmarks = 0;
            public readonly IDictionary<FacePart, IEnumerable<FacePoint>> Landmarks;

            public CandidateRectLandmarks(IDictionary<FacePart, IEnumerable<FacePoint>> landmarks)
            {
                Landmarks = landmarks;
            }
        }

        private class TopCandidates
        {
            public Rect Rect;
            public IDictionary<FacePart, IEnumerable<FacePoint>> Landmarks;

            public TopCandidates(Rect rect, IDictionary<FacePart, IEnumerable<FacePoint>> landmarks)
            {
                Rect = rect;
                Landmarks = landmarks;
            }
        }

        public AccurateFaceDetection(ILogger? logger = null, Rect? attractingRoi = null)
        {
            _attractingRoi = attractingRoi;
            var consideredFaceParts =
                logger ??= new LoggerFactory().CreateLogger<NullLogger>();
            _logger = logger;
            _hc = new HcFaceDetection(new HcFaceDetectionSettings());
            _recog = FaceRecognition.Create(new FaceRecognitionModelSettings().Location);
            //var lck = PythonEngine.AcquireLock();
            //_faceQNet = new FaceQNet(@"D:\Workspace\git_repos\FaceQnet\FaceQnet.h5");
            //PythonEngine.ReleaseLock(lck);
        }

        private FaceCoords GetFaceCoords(IDictionary<FacePart, IEnumerable<FacePoint>> landmarks)
        {
            var leftEye = landmarks[FacePart.LeftEye]
                .First(v => v.Point.X == landmarks[FacePart.LeftEye].Min(p => p.Point.X));
            var rightEye = landmarks[FacePart.RightEye]
                .First(v => v.Point.X == landmarks[FacePart.RightEye].Max(p => p.Point.X));
            var leftCheek = landmarks[FacePart.Chin]
                .First(v => v.Point.X == landmarks[FacePart.Chin].Min(p => p.Point.X));
            var rightCheek = landmarks[FacePart.Chin]
                .First(v => v.Point.X == landmarks[FacePart.Chin].Max(p => p.Point.X));
            var downChin = landmarks[FacePart.Chin]
                .First(v => v.Point.Y == landmarks[FacePart.Chin].Max(p => p.Point.Y));
            return new FaceCoords(leftEye.Point.X, leftEye.Point.Y, leftCheek.Point.X, leftCheek.Point.Y, downChin.Point.X, downChin.Point.Y, rightEye.Point.X,
                rightEye.Point.Y, rightCheek.Point.X, rightCheek.Point.Y);
        }

        private List<IDictionary<FacePart, IEnumerable<FacePoint>>> FindLandmarks(Image image)
        {
            if (_attractingRoi.HasValue)
            {
                var largeRoi = Rect.Inflate(_attractingRoi.Value, 2, 2);
                var landmarks = _recog
                    .FaceLandmark(image, new []
                    {
                        new Location(largeRoi.Left, largeRoi.Top, largeRoi.Right, largeRoi.Bottom)
                    }).ToList();
                return landmarks;
            }
            else
            {
                var landmarks = _recog
                    .FaceLandmark(image).ToList();
                return landmarks;
            }


        }

        private void SetPreviousOutput(FaceDetectionOutput output)
        {
            _prevOutput = output;
            if (_attractingRoi.HasValue)
            {
                _attractingRoi = output.faceRect;
            }
            _totalProcessed++;
            _areaSum += CalcArea(output.faceRect);
            _avgArea = _areaSum / _totalProcessed;
        }

        private double CalcArea(Rect rect)
        {
            return rect.Width * rect.Height;
        }

        private bool IntersectWith(Rect rect, Rect rect2)
        {
            if (_prevOutput == null)
            {
                return true;
            }
            return rect2.IntersectsWith(rect) || rect2.Contains(rect) || rect.Contains(rect2);
        }

        private TopCandidates? SelectBasedOnAreaAndPos(TopCandidates selected, TopCandidates[] topCandidates)
        {
            var smallerArea = CalcArea(selected.Rect) < _avgArea / 2.0;
            var biggerArea = CalcArea(selected.Rect) > _avgArea * 1.5;

            if (_attractingRoi.HasValue)
            {
                topCandidates = topCandidates.Where(c => IntersectWith(c.Rect, _attractingRoi.Value)).ToArray();
            }

            if (_totalProcessed > 0 && biggerArea)
            {
                _logger.LogDebug("Selecting another rect due to bigger area value");

                var newSelected = topCandidates.FirstOrDefault(arg => CalcArea(arg.Rect) < _avgArea * 1.5);
                return newSelected;
            }

            if (_totalProcessed > 0 && smallerArea)
            {
                _logger.LogDebug("Selecting another rect due to smaller area value");

                var newSelected = topCandidates.FirstOrDefault(arg => CalcArea(arg.Rect) > _avgArea / 2.0);
                return newSelected;
            }

            return selected;
        }

        private FaceDetectionOutput? DetectFacesFromRects(Mat image, Image dnImage, Rect[] rects)
        {
            var totalLandmarks = FindLandmarks(dnImage);

            _logger.LogDebug("HC Detected {Count} rects", rects.Length);
            _logger.LogDebug("Recog Detected {Count} coords", totalLandmarks.Count);

            var candidates = new Dictionary<Rect, CandidateRectLandmarks>();

            foreach (var landmarks in totalLandmarks.Where(l =>
                l.Keys.Intersect(ConsideredFaceParts).All(ConsideredFaceParts.Contains)))
            {
                foreach (var rect in rects)
                {
                    foreach (var (k, v) in landmarks.Where((kv) => ConsideredFaceParts.Contains(kv.Key)))
                    {
                        var intersectLandmarks = v.Count(p => rect.Contains(p.Point.X, p.Point.Y));
                        if (candidates.ContainsKey(rect))
                        {
                            candidates[rect].IntersectLandmarks += intersectLandmarks;
                        }
                        else
                        {
                            candidates[rect] = new CandidateRectLandmarks(landmarks);
                        }
                    }
                }
            }

            if (candidates.Count == 0)
            {
                _logger.LogWarning("0 rect candidates");
                return null;
            }

            // var cl = image.Clone();
            // foreach (var rect in candidates.Keys)
            // {
            //     Cv2.Rectangle(cl, rect, Scalar.Black);
            // }
            // Cv2.ImShow("framerec", cl);
            // Cv2.WaitKey(1);


            var topCandidates = candidates.OrderByDescending(kv => kv.Value.IntersectLandmarks)
                .Select(kv => new TopCandidates(kv.Key, kv.Value.Landmarks)).ToArray();
            _logger.LogDebug("Selected {Count} candidate rects", topCandidates.Length);

            

            // var topCandidatesFaceQNet = topCandidates
            //     .Select(v =>
            //     {
            //         var mat = new Mat(image, v.Rect).Clone();
            //         Cv2.Resize(mat, mat, new Size(96,96));
            //         Cv2.CvtColor(mat, mat, ColorConversionCodes.BGRA2RGB);
            //
            //         var sc = _faceQNet.GetScore(mat);
            //         //Cv2.ImShow("f", mat);
            //         //Cv2.WaitKey();
            //         return new {RectLand = v, Image = mat};
            //     }).OrderByDescending(v => _faceQNet.GetScore(v.Image)).ToArray();
            //
            // var selected = topCandidatesFaceQNet[0].RectLand;

            //select from top intersect
            var selected = topCandidates[0];
            //select from area change
            selected = SelectBasedOnAreaAndPos(selected, topCandidates);

            if (selected == null)
            {
                return null;
            }

            var output = new FaceDetectionOutput(selected.Rect, GetFaceCoords(selected.Landmarks), selected.Landmarks);
            SetPreviousOutput(output);
            return output;
        }

        public FaceDetectionOutput? DetectFace(Mat image)
        {
            var rects = _hc.DetectFrontalThenProfileFaces(image);

            if (!bufInitialized)
            {
                buff = new byte[image.Rows * image.Cols * image.ElemSize()];
                bufInitialized = true;
            }

            using var dnImage = LoadImage(image, buff);
            var output = DetectFacesFromRects(image, dnImage, rects);

            if (output == null)
            {
                _logger.LogWarning("Falling back to CNN detector");
                rects = _recog.FaceLocations(dnImage)
                    .Select(v => new Rect(v.Left, v.Top, v.Right - v.Left, v.Bottom - v.Top)).ToArray();
                output = DetectFacesFromRects(image, dnImage, rects);
            }

            if (output == null)
            {
                _logger.LogWarning("Returning previous output");
                return _prevOutput;
            }

            return output;
        }

        public void Dispose()
        {
            _hc.Dispose();
        }
    }
}