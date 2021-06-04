using System;
using System.IO;
using System.Runtime.InteropServices;
using FaceRecognitionDotNet;
using OpenCvSharp;

namespace Common
{
    public static class Utils
    {
        public static Image LoadImage(Mat photo, byte[] bytes = null)
        {
            if (bytes == null)
            {
                bytes = new byte[photo.Width * photo.Height * photo.Channels()];
            }

            Marshal.Copy(photo.Data, bytes, 0, bytes.Length);
            var img = FaceRecognition.LoadImage(bytes, photo.Rows, photo.Cols, (int)photo.Step(), Mode.Rgb);
            return img;
        }

        public static void EnsureEmpty(string dirpath)
        {
            if (Directory.Exists(dirpath))
            {
                Console.WriteLine("Removing " + dirpath);
                Directory.Delete(dirpath, true);
            }

            Directory.CreateDirectory(dirpath);
        }
    }
}