using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Common;
using Konsole;
using MediaToolkit;
using MediaToolkit.Model;
using MediaToolkit.Options;
using OpenCvSharp;
using Sharprompt;
using static Common.Utils;

namespace DataPreprocessing
{
    class FrameExtractStep
    {
        public void Execute(string videoFile, string baseDir, int freq = 300, double valSplit = 0)
        {
            var framesDir = Path.Combine(baseDir, "frames");
            EnsureEmpty(framesDir);
            using var engine = new Engine();
            var mp4 = new MediaFile { Filename = videoFile };

            engine.GetMetadata(mp4);

            var n = 0;
            var total = (int)Math.Floor(mp4.Metadata.Duration.TotalMilliseconds / freq);
            var trainCount = (int)((1.0 - valSplit) * total);
            var valCount = (int)(valSplit * total);

            var remaining = total - (trainCount + valCount);
            trainCount += remaining;

            var pb = new ProgressBar(total);
            Console.WriteLine($"Saving {total} frames");

            var trainDir = Path.Combine(framesDir, "training");
            EnsureEmpty(trainDir);
            var valiDir = Path.Combine(framesDir, "validation");
            if (valCount > 0)
            {
                EnsureEmpty(valiDir);
            }
            else
            {
                if (Directory.Exists(valiDir))
                {
                    Console.WriteLine("Removing old " + valiDir);
                    Directory.Delete(valiDir, true);
                }
            }

            Console.WriteLine($"Reading {total} total frames");
            Console.WriteLine($"Initial training: {trainCount}");
            Console.WriteLine($"Initial validation: {valCount}");
            if (!Prompt.Confirm("continue"))
            {
                return;
            }

            int i = 0;
            while (i < mp4.Metadata.Duration.TotalMilliseconds)
            {
                var options = new ConversionOptions { Seek = TimeSpan.FromMilliseconds(i) };

                if (trainCount > 0)
                {
                    var outputFile = new MediaFile
                        { Filename = Path.Combine(framesDir, "training", $"{n++}.jpeg") };
                    engine.GetThumbnail(mp4, outputFile, options);
                    trainCount--;
                }else if (valCount > 0)
                {
                    var outputFile = new MediaFile
                        { Filename = Path.Combine(framesDir, "validation", $"{n++}.jpeg") };
                    engine.GetThumbnail(mp4, outputFile, options);
                    valCount--;
                }

                i += freq;
                pb.Refresh(n, $"Processed frame {n}");
            }
        }
    }
}