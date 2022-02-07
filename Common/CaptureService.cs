using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using OpenCvSharp;

namespace Common
{
    public class CaptureService
    {
        private static bool _isCapturing;
        private VideoCapture cap = null;

        public CaptureService(string? filePath = null, int? camIndex = null)
        {
            Debug.Assert(filePath != null ^ camIndex != null);
            if (filePath != null)
            {
                cap = VideoCapture.FromFile(filePath);
            }
            else
            {
                cap = VideoCapture.FromCamera(camIndex.Value);
            }
        }

        public bool IsCapturing => _isCapturing;

        public int TotalFrames => (int)cap.Get(VideoCaptureProperties.FrameCount);

        private void ValidateMat(Mat mat)
        {
            if (mat.Empty() || mat.Width <= 0 || mat.Height <= 0)
            {
                throw new Exception("Invalid photo");
            }

        }

        public Mat CaptureSingleFrame()
        {
            VideoCapture cap = new VideoCapture(0);
            var mat = cap.RetrieveMat();
            cap.Release();

            ValidateMat(mat);

            return mat;
        }

        public async IAsyncEnumerable<Mat> CaptureFrames([EnumeratorCancellation] CancellationToken ct)
        {
            _isCapturing = true;
            while (!ct.IsCancellationRequested)
            {
                //todo
                try
                {
                    await Task.Delay(34, ct);
                }
                catch (TaskCanceledException)
                {
                    break;
                }
                var frame = cap.RetrieveMat();
                if (frame != null && !frame.Empty())
                {
                    ValidateMat(frame);
                    yield return frame;
                }
                else
                {
                    break;
                }
            }


            cap.Release();
            _isCapturing = false;
        }
    }
}
