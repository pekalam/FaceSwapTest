using System;
using System.Linq;
using System.Threading;
using Common;
using FaceRecognitionDotNet;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Console;
using Microsoft.Extensions.Options;
using OpenCvSharp;
using static Common.Utils;

//PythonInit.Init();

// var recog = FaceRecognition.Create(new FaceRecognitionModelSettings().Location);
// var hc = new HcFaceDetection(new HcFaceDetectionSettings());
// var imgp = @"D:\Workspace\FaceSwapDataset\novak\frames\training\0.jpeg";
// var img = Cv2.ImRead(imgp);
//
// var rects = hc.DetectFrontalThenProfileFaces(img);
// var masked = MaskUtils.GetMasked(img, recog.FaceLandmark(LoadImage(img)).First());
// masked = new Mat(masked, rects[0]);
//
// var missingMask = Mat.Zeros(masked.Size(), MatType.CV_8U).ToMat();
//
// // Cv2.InRange(masked, new Scalar(0, 0, 0), new Scalar(0, 0, 0), missingMask);
//
// var gray = new Mat();
// Cv2.CvtColor(masked, gray, ColorConversionCodes.BGR2GRAY);
// var thresh = new Mat();
// Cv2.Threshold(gray, thresh, 0, 255, ThresholdTypes.BinaryInv);
//
// Cv2.GaussianBlur(thresh, thresh, new Size(5,5), 11.0);
//
// Cv2.ImShow("asdasd",thresh);
// Cv2.WaitKey();
//
// var imgp2 = @"D:\Workspace\FaceSwapDataset\serena\frames\training\0.jpeg";
// var img2 = Cv2.ImRead(imgp2);
//
// new Mat(img2, rects[0]).CopyTo(masked, thresh);
//
// Cv2.ImShow("asdasd", masked);
// Cv2.WaitKey();

var videoPath = @"D:\nole.mp4";
var capServcie = new CaptureService(filePath: videoPath);
var to = capServcie.TotalFrames;
AccurateFaceDetection accFaceDetection = null;

var en = capServcie.CaptureFrames(CancellationToken.None).GetAsyncEnumerator();
var colors = new Scalar[] { Scalar.Red, Scalar.Blue, Scalar.Green, Scalar.Pink};

int frameCount = 0;
Rect? attractingRoi;
while (true)
{
    frameCount++;

    var t = en.MoveNextAsync();

    t.AsTask().GetAwaiter().GetResult();
    using var mat = en.Current;

    if (frameCount == 1)
    {
        attractingRoi = Cv2.SelectROI(mat);
        accFaceDetection = new AccurateFaceDetection(new ConsoleLogger(), attractingRoi);
    }

    var faceDetectionOutput = accFaceDetection.DetectFace(mat);

    int i = 0;

    Cv2.Rectangle(mat, faceDetectionOutput.faceRect, colors[i++ % colors.Length]);

    Cv2.Circle(mat, faceDetectionOutput.coords.leftEyeX, faceDetectionOutput.coords.leftEyeY, 5, colors[i++ % colors.Length]);
    Cv2.Circle(mat, faceDetectionOutput.coords.rightEyeX, faceDetectionOutput.coords.rightEyeY, 5, colors[i++ % colors.Length]);
    Cv2.Circle(mat, faceDetectionOutput.coords.leftCheekX, faceDetectionOutput.coords.leftCheekY, 5, colors[i++ % colors.Length]);
    Cv2.Circle(mat, faceDetectionOutput.coords.rightCheekX, faceDetectionOutput.coords.rightCheekY, 5, colors[i++ % colors.Length]);
    Cv2.Circle(mat, faceDetectionOutput.coords.downChinX, faceDetectionOutput.coords.downChinY, 5, colors[i++ % colors.Length]);

    Console.WriteLine(frameCount);
    // Cv2.CvtColor(mat, mat, ColorConversionCodes.RGB2BGR);
    Cv2.ImShow("frame", mat);
    Cv2.WaitKey(1);
}