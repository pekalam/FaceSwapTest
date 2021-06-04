using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;
using Common;
using FaceRecognitionDotNet;
using OpenCvSharp;
using Sharprompt;
using static Common.Utils;
using Point = OpenCvSharp.Point;

namespace DataPreprocessing
{
    class ExtractFacePipeline
    {
        private string _facesDir;

        public ExtractFacePipeline(string facesDir)
        {
            _facesDir = facesDir;
        }

        public void Execute()
        {
            using var hc = new HcFaceDetection(new HcFaceDetectionSettings());
            using var recog = FaceRecognition.Create(new FaceRecognitionModelSettings().Location);

            var consideredFaceParts = new[]
                {FacePart.LeftEye, FacePart.RightEye, FacePart.Chin};

            foreach (var file in Directory.EnumerateFiles(_facesDir))
            {
                var mat = Cv2.ImRead(file);
                var img = LoadImage(mat);
                var landmarks = recog.FaceLandmark(img).ToList();
                var rects = hc.DetectFrontalThenProfileFaces(mat);

                // var tuple = rects.BiggestWithConsideredFaceParts(coords, consideredFaceParts);
                //
                //  if (!tuple.HasValue)
                //  {
                //      continue;
                //  }


                //var mask = new Mat(mat.Size(), MatType.CV_8U, new Scalar(255));

                if (landmarks.Count == 0)
                {
                    continue;
                }

                var landmark = landmarks[0];
                var lChin = landmark[FacePart.Chin]
                    .First(v => v.Index == 0).Point;
                var c = new List<Point>();

                c.Add(new Point(lChin.X, lChin.Y));

                foreach (var el in landmark[FacePart.LeftEyebrow].Where(v => v.Index < 20))
                {
                    c.Add(new Point(el.Point.X, Math.Max(el.Point.Y, 0)));
                }

                foreach (var el in landmark[FacePart.RightEyebrow].Where(v => v.Index >= 24))
                {
                    c.Add(new Point(el.Point.X, Math.Max(el.Point.Y, 0)));
                }

                if (landmark[FacePart.Chin].Max(v => v.Point.Y) < 96 - 6)
                {
                    Console.WriteLine("dropping");
                    continue;
                }
                else
                {
                    foreach (var el in landmark[FacePart.Chin].Reverse())
                    {
                        var cpt = new Point(el.Point.X, Math.Max(el.Point.Y, 0));
                        c.Add(cpt);
                    }
                }


                var newImg = mat.Clone();
                newImg.SetTo(Scalar.Black);
                Point[][] contours = new Point[][]
                {
                    c.ToArray()
                };
                Cv2.DrawContours(newImg, contours, 0, Scalar.White, -1);
                mat.CopyTo(newImg, newImg);

                Cv2.ImShow("ads", newImg);
                Cv2.WaitKey();
            }
        }
    }

    class Program
    {
        private const string BaseDir = @"D:\Workspace\FaceSwapDataset";
        private static readonly string P1Dir = Path.Combine(BaseDir, "novak");
        private static readonly string P2Dir = Path.Combine(BaseDir, "serena");

        private static string P1ocDir(string dsName) => Path.Combine(P1Dir, dsName, "oc");
        private static string P1facesDir(string dsName) => Path.Combine(P1Dir, dsName, "faces");
        private static string P2ocDir(string dsName) => Path.Combine(P2Dir, dsName, "oc");
        private static string P2facesDir(string dsName) => Path.Combine(P2Dir, dsName, "faces");

        private const int ThreadCount = 8;

        static async Task Main(string[] args)
        {
            PythonInit.Init();

            var actions = new[]
            {
                "Detect and normalize p1", "Detect and normalize p2", "Open closed extract p1", "Open closed extract p2", "Finalize", "Extract frames", "Info"
            };


            async Task DetectAndTransform(string personDir)
            {
                var trainingFrames = Path.Combine(personDir, "frames", "training");
                var validationFrames = Path.Combine(personDir, "frames", "validation");

                Task ExecPipeline(string framesDir, string dsName)
                {
                    var detectTranNormPipeline = new DetectAndNormPipeline(framesDir,
                        Path.Join(personDir, dsName), ThreadCount);
                    return detectTranNormPipeline.Execute();
                }

                await ExecPipeline(trainingFrames, "training");
                if (Directory.Exists(validationFrames))
                {
                    await ExecPipeline(validationFrames, "validation");
                }
            }

            async Task OpenedClosedExtractPipeline(Func<string, string> personFacesDir, Func<string, string> personOcDir)
            {
                var trainFaces = personFacesDir("training");
                var valFaces = personFacesDir("validation");

                var trainOc = personOcDir("training");
                var valOc = personOcDir("validation");

                Task ExecPipeline(string facesDir, string ocDir)
                {
                    var p = new OpenedClosedExtractPipeline(facesDir, ocDir, ThreadCount);
                    return p.Execute();
                }

                await ExecPipeline(trainFaces, trainOc); 
                if (Directory.Exists(valFaces))
                {
                    await ExecPipeline(valFaces, valOc);
                }
            }

            while (true)
            {
                var action = Prompt.Select("Select", actions);

                if (action == actions[0])
                {
                    await DetectAndTransform(P1Dir);
                }

                if (action == actions[1])
                {
                    await DetectAndTransform(P2Dir);
                }

                if (action == actions[2])
                {
                    await OpenedClosedExtractPipeline(P1facesDir, P1ocDir);
                }

                if (action == actions[3])
                {
                    await OpenedClosedExtractPipeline(P2facesDir, P2ocDir);
                }

                if (action == actions[4])
                {
                    Console.WriteLine("Finalizing training ds");
                    if (Directory.Exists(Path.Combine(BaseDir, "dataset")))
                    {
                        Console.WriteLine("Removing existing dataset");
                        Directory.Delete(Path.Combine(BaseDir, "dataset"), true);
                    }

                    var dsFin = new FinalizeDataset(BaseDir, Path.Join(P1Dir, "training"), Path.Join(P2Dir, "training"), "novak", "serena", "training");
                    dsFin.Finalize();

                    //assuming directories exist for two persons
                    if (Directory.Exists(Path.Join(P1Dir, "validation")))
                    {
                        dsFin = new FinalizeDataset(BaseDir, Path.Join(P1Dir, "validation"), Path.Join(P2Dir, "validation"), "novak", "serena", "validation");
                        dsFin.Finalize();
                    }
                }

                if (action == actions[5])
                {
                    int freq = Prompt.Input<int>("Enter freq");

                    var extractFrames = new FrameExtractStep();
                    extractFrames.Execute(@"D:\nole.mp4", P1Dir,freq, 0.2);
                    extractFrames.Execute(@"D:\serena.mp4", P2Dir,freq, 0.2);
                    Console.WriteLine("Completed");
                }

                if (action == actions[6])
                {
                    var dsDir = Path.Combine(BaseDir, "dataset");
                    if (!Directory.Exists(dsDir))
                    {
                        return;
                    }

                    void ShowInfo(string trainDir, string valDir, string person)
                    {
                        int totalTrain = Directory.EnumerateFiles(trainDir).Count();
                        int totalTrainO = Directory
                            .EnumerateFiles(trainDir).Count(s => s.Split(Path.DirectorySeparatorChar)[^1].Contains('o'));
                        int totalTrainC = Directory
                            .EnumerateFiles(trainDir).Count(s => s.Split(Path.DirectorySeparatorChar)[^1].Contains('c'));
                        int totalWarped = Directory
                            .EnumerateFiles(trainDir).Count(s => s.Split(Path.DirectorySeparatorChar)[^1].StartsWith('w'));
                        Console.WriteLine($"{person} training: {totalTrain} o/c: {totalTrainO / (double)totalTrainC:F2} warped: {totalWarped}");

                        if (Directory.Exists(valDir))
                        {
                            int totalVal = Directory.EnumerateFiles(valDir).Count();
                            Console.WriteLine($"{person} validation: {totalVal}");
                            Console.WriteLine($"Val/Train: {totalVal / (double)totalTrain:F2}");
                        }
                    }

                    var trainingP1 = Path.Combine(dsDir, "novak", "training");
                    var validationP1 = Path.Combine(dsDir, "novak", "validation");
                    ShowInfo(trainingP1, validationP1, "novak");

                    var trainingP2 = Path.Combine(dsDir, "serena", "training");
                    var validationP2 = Path.Combine(dsDir, "serena", "validation");
                    ShowInfo(trainingP2, validationP2, "serena");

                }

            }



            //var p = new OpenedClosedExtractPipeline(P1facesDir, P1ocDir, 4);
            //await p.Execute();

           

            Console.ReadKey();
        }
    }
}