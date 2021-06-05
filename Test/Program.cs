﻿using System;
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

            //var modelDir = @"C:\Users\Marek\source\repos\DeepLearning\FaceSwapProject\faceswap_autoencoder\__saves__";
            var modelDir =
                @"C:\Users\Marek\source\repos\DeepLearning\FaceSwapProject\FaceSwapAutoencoder_from_template\__saves__";
            var capture = new CaptureService(filePath: @"D:\nole.mp4");
            var en = capture.CaptureFrames(CancellationToken.None).GetAsyncEnumerator();

            en.MoveNextAsync().AsTask().GetAwaiter().GetResult();

            var frameSz = en.Current.Size();
            var faceLocation = Cv2.SelectROI(en.Current);

            en.DisposeAsync().GetAwaiter().GetResult();
            capture = new CaptureService(filePath: @"D:\nole.mp4");
            en = capture.CaptureFrames(CancellationToken.None).GetAsyncEnumerator();

            // var model = new FaceSwapAutoencoder.FaceSwapAutoencoder(Path.Combine(modelDir, "model2.pb"), faceLocation);
            var model = new FaceSwapAutoencoder.FaceSwapAutoencoder(Path.Combine(modelDir, "model_masked1.pb"), faceLocation);


            // var en = capture.CaptureFrames(CancellationToken.None, @"D:\nole.mp4").GetAsyncEnumerator();


            FaceSwapAutoencoder.FaceSwapAutoencoder.Output prevOutput = null;
            int p = 0;

            bool writeFile = false;
            VideoWriter? writer = null;

            if(writeFile) writer = new VideoWriter("output.avi", FourCC.MJPG, 25, frameSz);

            var totalFrames = capture.TotalFrames;
            while (true)
            {
                var t = en.MoveNextAsync();
                t.AsTask().GetAwaiter().GetResult();
                using var mat = en.Current;

                p++;
                if (p < 370)
                {
                    //continue;
                }

                var modelOut = model.Call(mat, true);
                if (modelOut == null)
                {
                    modelOut = prevOutput;
                }
                Debug.Assert(modelOut != null);
                var (r1, r2, preprocessedOutput) = modelOut;
                var faceRect = preprocessedOutput.faceRect;
                var target = r1;
                //var target = r1;

                using var output = model.Preprocessing.InverseAffine(np.squeeze(target * 255.0f), preprocessedOutput);
                Cv2.CvtColor(output, output, ColorConversionCodes.RGB2BGR);
                output.ConvertTo(output, MatType.CV_8UC3);
                Cv2.Resize(output, output, new Size(faceRect.Width, faceRect.Height), interpolation: InterpolationFlags.Lanczos4);

                using var orgOutput = output.Clone();
                using var orgFace = new Mat(mat, faceRect);
                using var orgFaceCpy = orgFace.Clone();

                using var missingMask = Mat.Zeros(output.Size(), MatType.CV_8U).ToMat();
                //Cv2.BitwiseNot(missingMask, missingMask);


                using var gray = new Mat();
                Cv2.CvtColor(output, gray, ColorConversionCodes.BGR2GRAY);
                Cv2.Threshold(gray, missingMask, 0, 255, ThresholdTypes.BinaryInv);

                Cv2.GaussianBlur(missingMask, missingMask, new Size(9, 9), 16.0);

                //missing part
                orgFace.CopyTo(output, missingMask);

                //copy face
                output.CopyTo(new Mat(mat, faceRect));

                prevOutput = modelOut;


                var orgPreviewRect = new Rect(0, 0, orgFaceCpy.Width, orgFaceCpy.Height);
                orgFaceCpy.CopyTo(new Mat(mat, orgPreviewRect));
                orgPreviewRect.Y += orgPreviewRect.Height;

                orgOutput.CopyTo(new Mat(mat, orgPreviewRect));

                Cv2.ImShow("frame", mat);
                Cv2.WaitKey(5);

                if (writeFile)
                {
                    writer.Write(mat);
                }
                Console.WriteLine($"{p}/{totalFrames}");
            }
        }
    }
}