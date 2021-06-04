using System;
using System.Linq;
using System.Threading;
using Common;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Console;
using Microsoft.Extensions.Options;
using OpenCvSharp;
using static Common.Utils;

//PythonInit.Init();

var videoPath = @"D:\serena.mp4";
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
    Cv2.CvtColor(mat, mat, ColorConversionCodes.RGB2BGR);
    Cv2.ImShow("frame", mat);
    Cv2.WaitKey(1);
}