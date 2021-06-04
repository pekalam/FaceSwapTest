using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Enumeration;
using System.Linq;
using System.Threading;
using Common;
using FaceRecognitionDotNet;
using OpenCvSharp;
using Python.Runtime;
using static Common.Utils;
using Point = OpenCvSharp.Point;

namespace DataPreprocessing
{
    public class DetectAndNormalizeStep
    {
        private readonly string? _tempPath;
        private object _faceLck = new object();
        private object _initLck = new();
        object pbLck = new object();
        public int processed = 0;
        public int totalFaces = 0;
        public int totalMasked = 0;
        public int dropped = 0;
        public int maskedDropped = 0;
        const int FaceW = 96;
        const int FaceH = 96;
        const float FACEQ_ACCEPTED_SCORE = 0.20f;
        const string FACENETQ_PATH = @"D:\Workspace\git_repos\FaceQnet\FaceQnet_v1.h5";
        private string droppedDir;
        private string saveDir;

        public DetectAndNormalizeStep(string personDir, string? tempPath = null)
        {
            _tempPath = tempPath;
            saveDir = Path.Combine(personDir, "faces");
            droppedDir = Path.Combine(personDir, "faces", "dropped");
            EnsureEmpty(saveDir);
            EnsureEmpty(droppedDir);
        }

        public (FrameReadStep.ReadFrame[] readFrames, string faceQNetPath) PrepareFaceqNetStepFunc(
            FrameReadStep.ReadFrame[] readFrames)
        {
            string path;
            if (_tempPath != null)
            {
                path = Path.Combine(_tempPath, Guid.NewGuid().ToString());
            }
            else
            {
                path = Path.GetTempFileName();
            }

            File.Copy(FACENETQ_PATH, path, true);
            return (readFrames, path);
        }

        private Mat? GetMasked(Mat img, Rect rect, IDictionary<FacePart, IEnumerable<FacePoint>> landmark)
        {
            var lChin = landmark[FacePart.Chin]
                .First(v => v.Index == 0).Point;
            var c = new List<Point>();

            c.Add(new Point(lChin.X, lChin.Y));

            foreach (var el in landmark[FacePart.LeftEyebrow].Where(v => v.Index < 20))
            {
                c.Add(new Point(el.Point.X, el.Point.Y));
            }

            foreach (var el in landmark[FacePart.RightEyebrow].Where(v => v.Index >= 24))
            {
                c.Add(new Point(el.Point.X, el.Point.Y));
            }

            if (landmark[FacePart.Chin].Max(v => v.Point.Y) < 96 - 6)
            {
                return null;
            }

            foreach (var el in landmark[FacePart.Chin].Reverse())
            {
                var cpt = new Point(el.Point.X, el.Point.Y);
                c.Add(cpt);
            }

            var newImg = img.Clone();
            newImg.SetTo(Scalar.Black);
            Point[][] contours = new Point[][]
            {
                c.ToArray()
            };
            Cv2.DrawContours(newImg, contours, 0, Scalar.White, -1);
            img.CopyTo(newImg, newImg);


            return newImg;
        }

        public void StepFunc((FrameReadStep.ReadFrame[] readFrames, string faceQNetPath) args)
        {
            AccurateFaceDetection accDetect = null;
            HeadPoseNormalization norm;
            lock (_initLck)
            {
                norm = new HeadPoseNormalization();
                accDetect = new AccurateFaceDetection();
            }

            FaceQNet faceQNet = null;
            lock (_faceLck)
            {
                var lck = PythonEngine.AcquireLock();
                faceQNet = new FaceQNet(args.faceQNetPath);
                PythonEngine.ReleaseLock(lck);
            }


            int i = 0;
            foreach (var read in args.readFrames)
                using (read.frame)
                {
                    var frame = read.frame;
                    var sep = read.fileName.Split(Path.DirectorySeparatorChar)[^1].Split('.');
                    var (orgFilename, orgExtension) = (sep[0], sep[^1]);

        

                    var output = accDetect.DetectFace(frame);

                    if (output != null)
                    {
                        var (landmarks, rect) = (output.landmarks, output.faceRect);

                        i++;
                        var masked = GetMasked(frame, rect, landmarks);
                        if (masked == null)
                        {
                            Interlocked.Add(ref maskedDropped, 1);
                            throw new Exception("dropped masked");
                        }
                        else
                        {
                            var normMasked = norm.NormalizePosition(masked, rect, landmarks).Item1;
                            Cv2.Resize(normMasked, normMasked, new Size(FaceW, FaceH));
                            masked.Dispose();
                            masked = normMasked;
                        }

                        using var fc = norm.NormalizePosition(frame, rect, landmarks).Item1;
                        var zsc = faceQNet.GetScore(fc);
                        Cv2.Resize(fc, fc, new Size(FaceW, FaceH));
                        var fileName = orgFilename + "-" + i + "." + orgExtension;
                        var maskedfileName = orgFilename + "-" + i + ".masked." + orgExtension;
                        if (zsc < FACEQ_ACCEPTED_SCORE)
                        {
                            Interlocked.Add(ref dropped, 1);
                            Interlocked.Add(ref maskedDropped, 1);
                            Cv2.ImWrite(Path.Combine(droppedDir, fileName), fc);
                            if (masked != null)
                            {
                                Cv2.ImWrite(Path.Combine(droppedDir, maskedfileName), masked);
                            }
                        }
                        else
                        {
                            Interlocked.Add(ref totalFaces, 1);
                            Cv2.ImWrite(Path.Combine(saveDir, fileName), fc);
                            if (masked != null)
                            {
                                Interlocked.Add(ref totalMasked, 1);
                                Cv2.ImWrite(Path.Combine(saveDir, maskedfileName), masked);
                            }
                        }

                        if (masked != null)
                        {
                            masked.Dispose();
                        }
                    }

                    lock (pbLck)
                    {
                        processed++;
                        //pb.Refresh(processed, $"Total preprocessed frames: {processed}");
                    }
                }


            Console.WriteLine($"Step created {i} images. Dropped {dropped}. Dropped masked {maskedDropped}");
        }
    }
}