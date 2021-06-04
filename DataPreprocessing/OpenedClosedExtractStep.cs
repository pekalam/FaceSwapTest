using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Common;
using FaceRecognitionDotNet;
using FaceRecognitionDotNet.Extensions;
using OpenCvSharp;
using Point = FaceRecognitionDotNet.Point;
using static Common.Utils;

namespace DataPreprocessing
{
    public class OpenedClosedExtractStep
    {
        private readonly string _personOcDir;
        private readonly string _eyesDirOpened;
        private readonly string _eyesDirClosed;
        private readonly string _eyesDirDropped;
        private readonly string _mouthDirOpened;
        private readonly string _mouthDirClosed;
        private readonly string _mouthDirDropped;
        const double EarThresh = 0.25d;
        const double EarThreshMoth = 0.76d;

        public OpenedClosedExtractStep(string personOcDir)
        {
            _personOcDir = personOcDir;
            _eyesDirOpened = Path.Combine(_personOcDir, "eyes", "opened");
            _eyesDirClosed = Path.Combine(_personOcDir, "eyes", "closed");
            _eyesDirDropped = Path.Combine(_personOcDir, "eyes", "dropped");
            _mouthDirOpened = Path.Combine(_personOcDir, "mouth", "opened");
            _mouthDirClosed = Path.Combine(_personOcDir, "mouth", "closed");
            _mouthDirDropped = Path.Combine(_personOcDir, "mouth", "dropped");
            EnsureEmpty(_eyesDirOpened);
            EnsureEmpty(_eyesDirClosed);
            EnsureEmpty(_eyesDirDropped);
            EnsureEmpty(_mouthDirOpened);
            EnsureEmpty(_mouthDirClosed);
            EnsureEmpty(_mouthDirDropped);
        }


        private double Euclidean(Point p1, Point p2)
        {
            return Math.Sqrt(Math.Pow(p1.X - p2.X, 2) + Math.Pow(p1.Y - p2.Y, 2));
        }

        private double CompEar(FacePart eye, IDictionary<FacePart, IEnumerable<FacePoint>> landmarks, Point[] points = null, Mat mat = null)
        {
            if (points == null)
            {
                points = landmarks.Where(l => l.Key == eye)
                    .SelectMany(kv => kv.Value).Select(v => v.Point).ToArray();
            }

  

            //foreach (var pt in points)
            //{
            //    Cv2.Circle(imgMat, pt.X, pt.Y, 1, Scalar.Aqua);
            //}

            var minY1 = points.OrderBy(p => p.Y)
                .First(p => p.X > points.Min(p => p.X) && p.X < points.Max(p => p.X));
            var minY2 = points.OrderBy(p => p.Y)
                .Where(p => p.X > points.Min(p => p.X) && p.X < points.Max(p => p.X)).ElementAt(1);

            var p2 = minY1.X < minY2.X ? minY1 : minY2;
            var p3 = minY1.X > minY2.X ? minY1 : minY2;

            //Cv2.Circle(imgMat, p2.X, p2.Y, 1, Scalar.Red);
            //Cv2.Circle(imgMat, p3.X, p3.Y, 1, Scalar.Yellow);

            var maxY1 = points.OrderByDescending(p => p.Y)
                .First(p => p.X > points.Min(p => p.X) && p.X < points.Max(p => p.X));
            var maxY2 = points.OrderByDescending(p => p.Y)
                .Where(p => p.X > points.Min(p => p.X) && p.X < points.Max(p => p.X)).ElementAt(1);

            var p6 = maxY1.X < maxY2.X ? maxY1 : maxY2;
            var p5 = maxY1.X > maxY2.X ? maxY1 : maxY2;

            //Cv2.Circle(imgMat, p6.X, p6.Y, 1, Scalar.Pink);
            //Cv2.Circle(imgMat, p5.X, p5.Y, 1, Scalar.Green);

            var p1 = points.OrderBy(p => p.X).First();
            var p4 = points.OrderBy(p => p.X).Last();

            //Cv2.Circle(imgMat, p1.X, p1.Y, 1, Scalar.Brown);
            //Cv2.Circle(imgMat, p4.X, p4.Y, 1, Scalar.Khaki);

            if (mat != null)
            {
                foreach (var point in new Point[] { p1, p2, p3, p4, p5, p6 })
                {
                    Cv2.Circle(mat, point.X, point.Y, 1, Scalar.Red);
                }
                Cv2.ImShow("f", mat);
                Cv2.WaitKey();

                var x = (Euclidean(p2, p6) + Euclidean(p3, p5)) / Euclidean(p1, p4) / 2.0d;
            }


            return (Euclidean(p2, p6) + Euclidean(p3, p5)) / Euclidean(p1, p4) / 2.0d;
        }


        public void StepFunc(FrameReadStep.ReadFrame[] read)
        {
            using var hc = new HcFaceDetection(new HcFaceDetectionSettings());
            using var recog = FaceRecognition.Create(new FaceRecognitionModelSettings().Location);
            recog.CustomEyeBlinkDetector = new EyeAspectRatioLargeEyeBlinkDetector(0.2, 0.2);

            int i = 0;
            int droppedCount = 0;
            byte[] buff = null;

            foreach (var readFrame in read.Where(x => !x.name.Contains("masked")))
                using (readFrame.frame)
                {
                    var imgMat = readFrame.frame;
                    if (buff == null)
                    {
                        buff = new byte[imgMat.Width * imgMat.Height * imgMat.Channels()];
                    }

                    Cv2.CvtColor(imgMat, imgMat, ColorConversionCodes.BGR2RGB);
                    using var img = LoadImage(imgMat, buff);
                    var alllandmarks = recog.FaceLandmark(img).ToList();
                    Cv2.CvtColor(imgMat, imgMat, ColorConversionCodes.RGB2BGR);

                    var opened = GetEarEyes(alllandmarks, recog);
                    var newName = readFrame.name + "-" + i + "." + readFrame.extension;
                    if (!opened.HasValue)
                    {
                        droppedCount++;
                        Cv2.ImWrite(Path.Combine(_eyesDirDropped, newName), imgMat);
                    }
                    else if (opened.Value)
                    {
                        Cv2.ImWrite(Path.Combine(_eyesDirOpened, newName), imgMat);
                    }
                    else
                    {
                        Cv2.ImWrite(Path.Combine(_eyesDirClosed, newName), imgMat);
                    }

                    var earMouth = GetEarMouth(alllandmarks, imgMat);
                    if (earMouth < 0)
                    {
                        droppedCount++;
                        Cv2.ImWrite(Path.Combine(_mouthDirDropped, newName), imgMat);
                    }
                    else if (earMouth >= EarThreshMoth)
                    {
                        Cv2.ImWrite(Path.Combine(_mouthDirOpened, newName), imgMat);
                    }
                    else
                    {
                        Cv2.ImWrite(Path.Combine(_mouthDirClosed, newName), imgMat);
                    }

                    i++;
                }
        }

        private bool? GetEarEyes(List<IDictionary<FacePart, IEnumerable<FacePoint>>> alllandmarks, FaceRecognition recog)
        {
            var consideredFaceParts = new[]
                {FacePart.LeftEye, FacePart.RightEye};

            var landmarks = alllandmarks.Where(d => d.Keys.Any(part => consideredFaceParts.Contains(part)))
                .Select(d =>
                    d.Where(kv => consideredFaceParts.Contains(kv.Key))
                        .ToDictionary(kv => kv.Key, kv => kv.Value)).FirstOrDefault();

            if (landmarks == null)
            {
                return null;
            }

            var left = CompEar(FacePart.LeftEye, landmarks);
            var right = CompEar(FacePart.RightEye, landmarks);
            var ratio = Math.Min(left, right);
            recog.EyeBlinkDetect(landmarks, out var leftBlink, out var rightBlink);

            if (ratio >= EarThresh && (!leftBlink && !rightBlink))
            {
                return true;
            }

            return false;
        }


        private double GetEarMouth(List<IDictionary<FacePart, IEnumerable<FacePoint>>> alllandmarks, Mat mat = null)
        {
            var consideredFaceParts = new[]
                {FacePart.BottomLip, FacePart.TopLip};

            var landmarks = alllandmarks.Where(d => d.Keys.Any(part => consideredFaceParts.Contains(part)))
                .Select(d =>
                    d.Where(kv => consideredFaceParts.Contains(kv.Key))
                        .ToDictionary(kv => kv.Key, kv => kv.Value)).FirstOrDefault();

            if (landmarks == null)
            {
                return -1;
            }


            var points = landmarks.Where(l => consideredFaceParts.Contains(l.Key))
                .SelectMany(kv => kv.Value).Select(v => v.Point).ToArray();



            var left = CompEar(FacePart.BottomLip, landmarks, points);

            return left;
        }
    }
}