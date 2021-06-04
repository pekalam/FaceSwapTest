using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Common;
using FaceRecognitionDotNet;
using FaceSwapAutoencoder;
using NumSharp;
using OpenCvSharp;
using Point = OpenCvSharp.Point;

// static Mat NumpyToMat(NDArray t)
// {
//     var w = t.ToArray<float>();
//
//     var wMat = new Mat(new[] { 96, 96 }, MatType.CV_32FC3);
//
//
//     for (int i = 0; i < 96; i++)
//     {
//         for (int j = 0; j < 96; j++)
//         {
//             Vec3f v = new Vec3f();
//
//             for (int k = 0; k < 3; k++)
//             {
//                 v[k] = w[j * 96 * 3 + i * 3 + k] * 255f;
//             }
//
//             wMat.Set(j, i, v);
//         }
//     }
//
//     Cv2.ImWrite("test.jpg", wMat);
//
//     wMat.ConvertTo(wMat, MatType.CV_8UC3);
//
//     return wMat;
// }


namespace Test
{
    class Program
    {
        static double RectDist(Rect r1, Rect r2)
        {
            return Math.Sqrt(Math.Pow(r1.X - r2.X, 2) + Math.Pow(r1.Y - r2.Y,2));
        }

        static Image LoadImage(Mat photo, byte[] bytes = null)
        {
            if (bytes == null)
            {
                bytes = new byte[photo.Width * photo.Height * photo.Channels()];
            }

            Marshal.Copy(photo.Data, bytes, 0, bytes.Length);
            var img = FaceRecognition.LoadImage(bytes, photo.Rows, photo.Cols, (int)photo.Step(), Mode.Rgb);
            return img;
        }

        static Mat? WarpToMatchLandmarks(Mat frame, Rect faceRoi, Mat newFacae, List<IDictionary<FacePart, IEnumerable<FaceRecognitionDotNet.Point>>> faceLandmarks)
        {
            if (faceLandmarks == null || faceLandmarks.Count == 0)
            {
                return null;
            }
            var dfaceLandmarks = faceLandmarks[0];
            if (!dfaceLandmarks.ContainsKey(FacePart.LeftEye) || !dfaceLandmarks.ContainsKey(FacePart.RightEye) ||
                !dfaceLandmarks.ContainsKey(FacePart.BottomLip))
            {
                return null;

            }

            var newFaceLandmarks = SharedFaceRecognitionModel.Model
                .FaceLandmark(LoadImage(newFacae)).ToList();

            if (newFaceLandmarks == null || newFaceLandmarks.Count == 0)
            {
                return null;

            }

            var dnewFaceLandmarks = newFaceLandmarks[0];

            if (dnewFaceLandmarks.Count == 0 || !dnewFaceLandmarks.ContainsKey(FacePart.LeftEye) || !dnewFaceLandmarks.ContainsKey(FacePart.RightEye) ||
                !dnewFaceLandmarks.ContainsKey(FacePart.BottomLip))
            {
                return null;

            }



            var newleftEye = dnewFaceLandmarks[FacePart.LeftEye].OrderBy(p => p.Point.X).First();
            var newrightEye = dnewFaceLandmarks[FacePart.RightEye].OrderBy(p => p.Point.X).Last();
            var newbottomLip = dnewFaceLandmarks[FacePart.BottomLip].OrderBy(p => p.Point.Y).Last();
            var leftEye = dfaceLandmarks[FacePart.LeftEye].OrderBy(p => p.X).First();
            var rightEye = dfaceLandmarks[FacePart.RightEye].OrderBy(p => p.X).Last();
            var bottomLip = dfaceLandmarks[FacePart.BottomLip].OrderBy(p => p.Y).Last();

            leftEye = new FaceRecognitionDotNet.Point(leftEye.X - faceRoi.X, leftEye.Y - faceRoi.Y);
            rightEye = new FaceRecognitionDotNet.Point(rightEye.X - faceRoi.X, rightEye.Y - faceRoi.Y);
            bottomLip = new FaceRecognitionDotNet.Point(bottomLip.X - faceRoi.X, bottomLip.Y - faceRoi.Y);

            var m = Cv2.GetAffineTransform(new[] {new Point2f(newleftEye.Point.X, newleftEye.Point.Y), new Point2f(newrightEye.Point.X, newrightEye.Point.Y), new Point2f(newbottomLip.Point.X, newbottomLip.Point.Y)},
                new[] {new Point2f(leftEye.X, leftEye.Y), new Point2f(rightEye.X, rightEye.Y), new Point2f(bottomLip.X, bottomLip.Y)});
            var ret = Mat.Zeros(newFacae.Size(), newFacae.Type()).ToMat();
            Cv2.WarpAffine(newFacae, ret, m, newFacae.Size(), borderMode: BorderTypes.Transparent);
            return ret;
        }

        static async Task Main(string[] args)
        {
            // var syncCtx = new CustomSynchronizationContext();
            //TODO use async enum in console app
            // SynchronizationContext.SetSynchronizationContext(syncCtx);

            var modelDir = @"C:\Users\Marek\source\repos\DeepLearning\FaceSwapProject\faceswap_autoencoder\__saves__";
            var model = new FaceSwapAutoencoder.FaceSwapAutoencoder(Path.Combine(modelDir, "model2.pb"));
            var capture = new CaptureService(filePath: @"D:\serena.mp4");


            var en = capture.CaptureFrames(CancellationToken.None).GetAsyncEnumerator();
            // var en = capture.CaptureFrames(CancellationToken.None, @"D:\nole.mp4").GetAsyncEnumerator();



            

            Mat previousFace = null;
            Rect previousRect = Rect.Empty;
            PreprocessedOutput prevOutput = null;
            int p = 0;

            bool seamlessClone = false;

            while (true)
            {
                var t = en.MoveNextAsync();
                t.AsTask().GetAwaiter().GetResult();
                var mat = en.Current;

                p++;
                if (p < 370)
                {
                    //continue;
                }

                var (r1, r2, preprocessedOutput) = model.Call(mat);
                var faceRect = preprocessedOutput.faceRect;
                var landmarks = preprocessedOutput.landmarks;
                var target = r2;
                //var target = r1;

                Rect selectedRect;
                if (!faceRect.HasValue)
                {
                    if (previousRect == Rect.Empty)
                    {
                        continue;
                    }
                    faceRect = previousRect;
                }
                if (previousRect != Rect.Empty && (!previousRect.IntersectsWith(faceRect.Value) || RectDist(faceRect.Value, previousRect) > 40 || faceRect.Value.Width*faceRect.Value.Height < previousRect.Width*previousRect.Height/2))
                {
                    selectedRect = previousRect;
                }
                else
                {
                    selectedRect = previousRect = faceRect.Value;
                }

                Mat wMat, wMask = null;
                if (!preprocessedOutput.faceRect.HasValue)
                {
                    if (previousFace == null)
                    {
                        continue;
                    }
                    if (prevOutput == null)
                    {
                        continue;
                    }
                    preprocessedOutput = prevOutput;
                    wMat = previousFace;
                }
                else
                {
                    wMat = model.Preprocessing.InverseAffine(np.squeeze(target * 255.0f), preprocessedOutput);
                    Cv2.CvtColor(wMat, wMat, ColorConversionCodes.RGB2BGR);
                }
                wMat.ConvertTo(wMat, MatType.CV_8UC3);
                Cv2.Resize(wMat, wMat, new Size(selectedRect.Width, selectedRect.Height));
                previousFace = wMat;
                var orgFace = new Mat(mat, selectedRect).Clone();

                //Cv2.CvtColor(mat, mat, ColorConversionCodes.BGR2BGRA);

                var warped = wMat.Clone();
                Cv2.CvtColor(warped, warped, ColorConversionCodes.BGR2GRAY);
                Cv2.FindContours(warped, out var contours, out var _, RetrievalModes.External, ContourApproximationModes.ApproxNone);

                var mask = Mat.Zeros(wMat.Size(), MatType.CV_8U).ToMat();
                Cv2.DrawContours(mask, contours, 0, new Scalar(255), -1);

                Cv2.Erode(mask, mask, new Mat()); 
                //wMat.CopyTo(new Mat(mat, selectedRect), mask);

                var center = new Point(selectedRect.X + selectedRect.Width / 2,
                    selectedRect.Y + selectedRect.Height / 2);

                for (int i = 0; i < preprocessedOutput.p2.Length; i++)
                {
                    preprocessedOutput.p2[i].X += selectedRect.X;
                    preprocessedOutput.p2[i].Y += selectedRect.Y;
                }

                var h = Cv2.GetAffineTransform(preprocessedOutput.p2, preprocessedOutput.p1);
                var m1 = Mat.FromArray(center.X, center.Y, 1);
                m1.ConvertTo(m1, MatType.CV_64F);

                var ap = h.Multiply(m1).ToMat();
                var maskCenter = new Point((float)ap.At<double>(0, 0),(float) ap.At<double>(0, 1));

                // center.X = (int)ap.At<double>(0, 0);
                // center.Y = (int)ap.At<double>(0, 1);



                var orgMat = mat.Clone();

                maskCenter.X -= selectedRect.X;
                maskCenter.Y -= selectedRect.Y;

                var circleMask = new Mat(mask.Size(), MatType.CV_8U, new Scalar(255));
                Cv2.Circle(circleMask, maskCenter, (int)(wMat.Width/2.0f), Scalar.Black,-1);



                var missingMask = new Mat(mat.Size(), MatType.CV_8U, new Scalar(255));
                var negMask = new Mat();
                Cv2.BitwiseNot(mask, negMask);
                negMask.CopyTo(new Mat(missingMask, selectedRect));

                //copy face
                if (!seamlessClone)
                {
                    wMat.CopyTo(new Mat(mat, selectedRect), mask);
                }

                // Cv2.ImShow("c", corners);
                // Cv2.WaitKey();

                //Cv2.Circle(mat, maskCenter, 5, Scalar.Red);
                //wMat.CopyTo(new Mat(mat, selectedRect), ~circleMask);
                // var ttt = new Mat();
                // wMat.CopyTo(ttt, ~circleMask);
                // Cv2.ImShow("sss", ttt);
                // Cv2.WaitKey();

                try
                {
                    if (seamlessClone)
                    {
                        Cv2.SeamlessClone(wMat, mat, null, center, mat, SeamlessCloneMethods.NormalClone);
                    }
                }
                catch (OpenCVException e)
                {
                    Console.WriteLine(e);
                    continue;
                }

                //copy missing part
                orgMat.CopyTo(mat, missingMask);

                prevOutput = preprocessedOutput;


                var orgPreviewRect = new Rect(0, 0, orgFace.Width, orgFace.Height);
                orgFace.CopyTo(new Mat(mat, orgPreviewRect));
                orgPreviewRect.Y += orgPreviewRect.Height;

                wMat.CopyTo(new Mat(mat, orgPreviewRect));

                Cv2.ImShow("frame", mat);
                Cv2.WaitKey(5);
            }
        }
    }
}