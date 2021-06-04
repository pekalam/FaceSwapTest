using System.IO;
using OpenCvSharp;

namespace DataPreprocessing
{
    public class FrameReadStep
    {
        public record ReadFrame(Mat frame, string fileName, string name, string extension);

        public ReadFrame StepFunc(string fileName)
        {
            var sep = fileName.Split(Path.DirectorySeparatorChar)[^1].Split('.');
            var (orgFilename, orgExtension) = (sep[0], sep[^1]);
            if (sep.Length > 2)
            {
                for (int i = 1; i < sep.Length - 1; i++)
                {
                    orgFilename += sep[i];
                }
            }
            var mat = Cv2.ImRead(fileName);
            return new ReadFrame(mat, fileName, orgFilename, orgExtension);
        }
    }
}