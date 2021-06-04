using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Threading.Tasks.Dataflow;

namespace DataPreprocessing
{
    public class OpenedClosedExtractPipeline
    {
        private readonly string _facesDir;
        private readonly string _ocDir;
        private Task CompletionTask;
        private TransformBlock<string, FrameReadStep.ReadFrame> inputBlock;

        public OpenedClosedExtractPipeline(string facesDir, string ocDir, int threadCount)
        {
            _facesDir = facesDir;
            _ocDir = ocDir;
            var total = Directory.EnumerateFiles(facesDir).Count();
            var batch = total / threadCount;

            var frameReadStep = new FrameReadStep();
            var f1 = new TransformBlock<string, FrameReadStep.ReadFrame>(frameReadStep.StepFunc,
                new ExecutionDataflowBlockOptions
                {

                });
            var f1batch = new BatchBlock<FrameReadStep.ReadFrame>(batch);


            var openedClosedExtractStep = new OpenedClosedExtractStep(ocDir);
            var o1 = new ActionBlock<FrameReadStep.ReadFrame[]>(openedClosedExtractStep.StepFunc, 
                new ExecutionDataflowBlockOptions()
                {
                    MaxMessagesPerTask = threadCount,
                    MaxDegreeOfParallelism = threadCount,
                });


            f1.LinkTo(f1batch, new DataflowLinkOptions()
            {
                PropagateCompletion = true,
            });
            f1batch.LinkTo(o1, new DataflowLinkOptions()
            {
                PropagateCompletion = true,
            });

            inputBlock = f1;
            CompletionTask = o1.Completion;
        }

        public async Task Execute()
        {
            foreach (var file in Directory.EnumerateFiles(_facesDir))
            {
                inputBlock.Post(file);
            }
            inputBlock.Complete();

            await CompletionTask;
            Console.WriteLine("Complete");
        }
    }
}