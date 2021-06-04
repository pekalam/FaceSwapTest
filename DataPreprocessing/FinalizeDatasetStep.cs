using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;
using Sharprompt;
using static Common.Utils;

namespace DataPreprocessing
{
    class FinalizeDataset
    {
        private string _baseDir;
        private string _p1Dir;
        private string _p2Dir;
        private string _targetBaseDir;
        private readonly string _targetDirP1;
        private readonly string _targetDirP2;
        private const double DesiredOpenedClosedRatio = 1.3d;

        public FinalizeDataset(string baseDir, string p1Dir, string p2Dir, string p1Name, string p2Name, string dsName)
        {
            _baseDir = baseDir;
            _p1Dir = p1Dir;
            _p2Dir = p2Dir;
            _targetBaseDir = Path.Combine(_baseDir, "dataset");
            _targetDirP1 = Path.Combine(_targetBaseDir, p1Name, dsName);
            _targetDirP2 = Path.Combine(_targetBaseDir, p2Name, dsName);
            EnsureEmpty(_targetDirP1);
            EnsureEmpty(_targetDirP2);
        }

        static void RandomlyDistortImage(Mat image)
        {
            var rows = image.Rows;
            var cols = image.Cols;
            var rnd = new Random();
            var org = image.Clone();
            var p1 = new Point2f[] { new Point2f(0, 96), new Point2f(96, 96), new Point2f(96, 0) };
            var p2 = new Point2f[] { new Point2f(rnd.Next(0, 0), rnd.Next(98, 104)), new Point2f(rnd.Next(96, 96), rnd.Next(98, 104)), new Point2f(rnd.Next(98, 104), rnd.Next(-3, 0)) };
            p2[0].X = -(p2[^1].X - 96);
            var m = Cv2.GetAffineTransform(p1, p2);
            Cv2.Flip(image, image, FlipMode.Y);

            Cv2.WarpAffine(image, image, m, new Size(cols, rows));
            // Cv2.ImShow("x", image);
            // Cv2.WaitKey();
            // Cv2.ImShow("y", org);
            // Cv2.WaitKey();
        }

        private string MapToFacesDir(string filePath, string facesDir, bool masked)
        {
            var fileName = filePath.Split(Path.DirectorySeparatorChar)[^1];

            var extPart = fileName.Split('.');
            var name = extPart[0];
            var ext = extPart[^1];

            var faceName = string.Join("-", name.Split('-')[0..^1]);
            var faceFile = faceName + (masked ? ".masked" : "") + "." + ext;

            return Path.Combine(facesDir, faceFile);
        }

        private void CreateDataset(bool masked, string facesDir, string openedDir, string closedDir, int openedToAdd, int closedToAdd, string targetDir, int openedCount, int closedCount)
        {
            var rnd = new Random();
            string PopRandom(List<string> toPop)
            {
                var toPopItem = toPop[rnd.Next(0, toPop.Count)];
                toPop.Remove(toPopItem);

                return toPopItem;
            }

            var openedToCpy = openedCount + openedToAdd;
            var closedToCpy = closedCount + closedToAdd;

            var opened = Directory.EnumerateFiles(openedDir).Select(x => MapToFacesDir(x, facesDir, masked)).ToList();
            var (openedToWarp, openedCpy) = (opened.ToList(), opened.ToList());
            var closed = Directory.EnumerateFiles(closedDir).Select(x => MapToFacesDir(x, facesDir, masked)).ToList();
            var (closedToWarp, closedCpy) = (closed.ToList(), closed.ToList());

            int openedCopied = 0;
            int openedWarped = 0;
            while (openedCopied != openedToCpy)
            {
                if (opened.Count > 0)
                {
                    File.Copy(PopRandom(opened), Path.Combine(targetDir, $"{openedCopied}o.jpeg"));
                }
                else
                {
                    if (openedToWarp.Count == 0)
                    {
                        openedToWarp = openedCpy.ToList();
                    }

                    var randomWarped = PopRandom(openedToWarp);
                    using var img = Cv2.ImRead(randomWarped);
                    RandomlyDistortImage(img);
                    Cv2.ImWrite(Path.Combine(targetDir, $"w{openedWarped}_{openedCopied}o.jpeg"), img);
                }
                openedCopied++;
            }

            int closedCopied = 0;
            int closedWarped = 0;
            while (closedCopied != closedToCpy)
            {
                if (closed.Count > 0)
                {
                    File.Copy(PopRandom(closed), Path.Combine(targetDir, $"{closedCopied}c.jpeg"));
                }
                else
                {
                    if (closedToWarp.Count == 0)
                    {
                        closedToWarp = closedCpy.ToList();
                    }

                    var randomWarped = PopRandom(closedToWarp);
                    using var img = Cv2.ImRead(randomWarped);
                    RandomlyDistortImage(img);
                    Cv2.ImWrite(Path.Combine(targetDir, $"w{closedWarped}_{closedCopied}c.jpeg"), img);
                }
                closedCopied++;
            }
        }

        public void Finalize()
        {
            var p1OpenedDir = Path.Combine(_p1Dir, "oc", "eyes", "opened");
            var p1ClosedDir = Path.Combine(_p1Dir, "oc", "eyes", "closed");

            var p2OpenedDir = Path.Combine(_p2Dir, "oc", "eyes", "opened");
            var p2ClosedDir = Path.Combine(_p2Dir, "oc", "eyes", "closed");



            var canContinue = false;
            var desiredSize = 0;
            while (!canContinue)
            {
                var totalP1Opened = Directory.EnumerateFiles(p1OpenedDir).Count();
                var totalP1Closed = Directory.EnumerateFiles(p1ClosedDir).Count();
                var totalP2Opened = Directory.EnumerateFiles(p2OpenedDir).Count();
                var totalP2Closed = Directory.EnumerateFiles(p2ClosedDir).Count();

                var totalP1 = totalP1Opened + totalP1Closed;
                var totalP2 = totalP2Opened + totalP2Closed;

                desiredSize = Prompt.Input<int>($"Enter desired size of datasets (current p1: {totalP1} current p2: {totalP2})");

                var p1ToAdd = desiredSize - totalP1;
                var p1ClosedArtif = 0;
                var p1OpenedArtif = 0;
                if (p1ToAdd > 0)
                {
                    while (totalP1 != desiredSize)
                    {
                        if (totalP1Closed < totalP1Opened)
                        {
                            totalP1Closed++;
                            p1ClosedArtif++;
                        }
                        else
                        {
                            totalP1Opened++;
                            p1OpenedArtif++;
                        }
                        totalP1 = totalP1Opened + totalP1Closed;
                    }
                    Console.WriteLine($"{p1OpenedArtif + p1ClosedArtif} P1 elements must be added");
                }
                else if(p1ToAdd < 0)
                {
                    while (totalP1 != desiredSize)
                    {
                        if (totalP1Opened > totalP1Closed)
                        {
                            totalP1Opened--;
                        }
                        else
                        {
                            totalP1Closed--;
                        }
                        totalP1 = totalP1Opened + totalP1Closed;
                    }
                }

                var p1openedToAdd =
                    (DesiredOpenedClosedRatio * totalP1Closed - totalP1Opened) / (1 + DesiredOpenedClosedRatio);
                var p1closedToAdd =
                    (totalP1Opened + totalP1Closed) - totalP1Opened - p1openedToAdd - totalP1Closed;

                p1openedToAdd = (int)Math.Round(p1openedToAdd);
                p1closedToAdd = (int)Math.Round(p1closedToAdd);

                if (p1openedToAdd < 0)
                {
                    p1openedToAdd += p1OpenedArtif;
                }
                if (p1closedToAdd < 0)
                {
                    p1closedToAdd += p1ClosedArtif;
                }

                Console.WriteLine($"P1 opened to add: {p1openedToAdd} closed: {p1closedToAdd}");

                var p2ToAdd = desiredSize - totalP2;
                var p2ClosedArtif = 0;
                var p2OpenedArtif = 0;
                if (p2ToAdd > 0)
                {
                    while (totalP2 != desiredSize)
                    {
                        if (totalP2Closed < totalP2Opened)
                        {
                            totalP2Closed++;
                            p2ClosedArtif++;
                        }
                        else
                        {
                            totalP2Opened++;
                            p2OpenedArtif++;
                        }
                        totalP2 = totalP2Opened + totalP2Closed;
                    }
                    Console.WriteLine($"{p2OpenedArtif + p2ClosedArtif} P2 elements must be added");
                }

                var p2openedToAdd =
                    (DesiredOpenedClosedRatio * totalP2Closed - totalP2Opened) / (1 + DesiredOpenedClosedRatio);
                var p2closedToAdd =
                    (totalP2Opened + totalP2Closed) - totalP2Opened - p2openedToAdd - totalP2Closed;

                p2openedToAdd = (int)Math.Round(p2openedToAdd);
                p2closedToAdd = (int)Math.Round(p2closedToAdd);

                if (p2openedToAdd < 0)
                {
                    p2openedToAdd += p2OpenedArtif;
                }
                if (p2closedToAdd < 0)
                {
                    p2closedToAdd += p2ClosedArtif;
                }

                

                Console.WriteLine($"P2 opened to add: {p2openedToAdd} closed: {p2closedToAdd}");

                canContinue = Prompt.Confirm("OK?");

                if (canContinue)
                {
                    bool masked = Prompt.Confirm("masked");

                    CreateDataset(masked, Path.Combine(_p1Dir, "faces"), p1OpenedDir, p1ClosedDir, (int)p1openedToAdd, (int)p1closedToAdd, _targetDirP1, totalP1Opened, totalP1Closed);
                    CreateDataset(masked, Path.Combine(_p2Dir, "faces"), p2OpenedDir, p2ClosedDir, (int)p2openedToAdd, (int)p2closedToAdd, _targetDirP2, totalP2Opened, totalP2Closed);
                }
            }
            Console.WriteLine("Completed");
        }

    }
}
