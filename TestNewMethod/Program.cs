using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using Common;
using NumSharp;
using OpenCvSharp;

var modelDir =
    @"C:\Users\Marek\source\repos\DeepLearning\FaceSwapProject\FaceSwapAutoencoder_from_template\__saves__";

//var faceLocation = Cv2.SelectROI(en.Current);

var capture = new CaptureService(filePath: @"D:\nole.mp4");
var en = capture.CaptureFrames(CancellationToken.None).GetAsyncEnumerator();
en.MoveNextAsync().AsTask().GetAwaiter().GetResult();

// var model = new FaceSwapAutoencoder.FaceSwapAutoencoder(Path.Combine(modelDir, "model2.pb"), faceLocation);
var model = new FaceSwapAutoencoder.FaceSwapAutoencoder(Path.Combine(modelDir, "model_masked1.pb"));


var mat = en.Current;

var modelOut = model.Call(mat, true);
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
using var gray = new Mat();
Cv2.CvtColor(output, gray, ColorConversionCodes.BGR2GRAY);
Cv2.Threshold(gray, missingMask, 0, 255, ThresholdTypes.BinaryInv);

//Cv2.GaussianBlur(missingMask, missingMask, new Size(9, 9), 16.0);

//missing part
orgFace.CopyTo(output, missingMask);

//copy face
output.CopyTo(orgFace);

Cv2.BitwiseNot(missingMask, missingMask);
Cv2.FindContours(missingMask, out var contours, out var _, RetrievalModes.External, ContourApproximationModes.ApproxNone);
var topBoundingRect = contours.Select(Cv2.BoundingRect).OrderByDescending(r => r.Width * r.Height).First();
var topContour = contours.First(c => Cv2.BoundingRect((IEnumerable<Point>) c) == topBoundingRect);
var bestMask = Mat.Zeros(missingMask.Size(), missingMask.Type()).ToMat();
Cv2.DrawContours(bestMask, new List<IEnumerable<Point>>(){topContour}, -1, new Scalar(255, 255, 255), Cv2.FILLED);

var center = new Point(topBoundingRect.X + topBoundingRect.Width / 2, topBoundingRect.Y + topBoundingRect.Height / 2);
Cv2.SeamlessClone(orgFace, orgFaceCpy, bestMask, center, orgFaceCpy, SeamlessCloneMethods.NormalClone);

orgFaceCpy.CopyTo(orgFace);

Cv2.ImShow("mis", mat);
Cv2.WaitKey();
//
// Cv2.ImShow("as", orgFace);
// Cv2.WaitKey();
//
// Mat SeamlessClone(Mat missingMask, Mat output)
// {
//     Cv2.BitwiseNot(missingMask, missingMask);
//     Cv2.FindContours(missingMask, out var contours, out var _, RetrievalModes.External, ContourApproximationModes.ApproxNone);
//     Cv2.DrawContours(output, contours, -1, new Scalar(255, 255, 0));
//     Rect s = contours.Select(c => Cv2.BoundingRect(c)).OrderByDescending(b => b.Width * b.Height).First();
//     Cv2.SeamlessClone(orgFace, orgFaceCpy, missingMask, new Point(s.X + s.Width / 2, s.Y + s.Height / 2), orgFaceCpy, SeamlessCloneMethods.NormalClone);
// }

// Cv2.ImShow("output", orgFaceCpy);
// Cv2.WaitKey();