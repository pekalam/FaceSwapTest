using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Threading.Tasks.Dataflow;

namespace DataPreprocessing
{
    public class DetectAndNormPipeline
    {
        private readonly string _frameDir;
        private readonly string _personDir;
        private Task CompletionTask;
        private TransformBlock<string, FrameReadStep.ReadFrame> inputBlock;
        private DetectAndNormalizeStep detectAndNormStep;

        public DetectAndNormPipeline(string frameDir,string personDir, int threadCount)
        {
            _frameDir = frameDir;
            _personDir = personDir;
            var total = Directory.EnumerateFiles(frameDir).Count();
            var batch = total / threadCount;

            var frameReadStep = new FrameReadStep();
            var f1 = new TransformBlock<string, FrameReadStep.ReadFrame>(frameReadStep.StepFunc,
                new ExecutionDataflowBlockOptions
                {

                });
            var f1batch = new BatchBlock<FrameReadStep.ReadFrame>(batch);

            detectAndNormStep = new DetectAndNormalizeStep(personDir, @"D:\WIN10_SSDSamsung\TEMP\tempo");
            var d1 = new TransformBlock<FrameReadStep.ReadFrame[], (FrameReadStep.ReadFrame[], string)>(detectAndNormStep
                .PrepareFaceqNetStepFunc, new ExecutionDataflowBlockOptions()
            {
                MaxMessagesPerTask = threadCount,
                MaxDegreeOfParallelism = threadCount,
            });
            var d2 = new ActionBlock<(FrameReadStep.ReadFrame[], string)>(detectAndNormStep.StepFunc, new ExecutionDataflowBlockOptions()
            {
                MaxMessagesPerTask = threadCount,
                MaxDegreeOfParallelism = threadCount,
            });

            f1.LinkTo(f1batch, new DataflowLinkOptions()
            {
                PropagateCompletion = true,
            });
            f1batch.LinkTo(d1, new DataflowLinkOptions()
            {
                PropagateCompletion = true,
            });
            d1.LinkTo(d2, new DataflowLinkOptions()
            {
                PropagateCompletion = true,
            });

            CompletionTask = d2.Completion;
            inputBlock = f1;
        }

        public async Task Execute()
        {
            foreach (var file in Directory.EnumerateFiles(_frameDir))
            {
                inputBlock.Post(file);
            }
            inputBlock.Complete();

            await CompletionTask;
            Console.WriteLine("Complete");

            Console.WriteLine($"Created {detectAndNormStep.totalFaces} images masked: {detectAndNormStep.totalMasked}");
            Console.WriteLine($"Dropped {detectAndNormStep.dropped} images masked: {detectAndNormStep.maskedDropped}");
        }
    }
}