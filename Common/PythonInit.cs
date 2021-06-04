using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Python.Runtime;

namespace Common
{
    public static class PythonInit
    {
        public static void Init()
        {
            PythonEngine.PythonHome = @"C:\Users\Marek\miniconda3\envs\ml1";
            PythonEngine.Initialize();
            string codeToRedirectOutput =
                "import sys\n" +
                "from io import StringIO\n" +
                "sys.stdout = mystdout = StringIO()\n" +
                "sys.stdout.flush()\n" +
                "sys.stderr = mystderr = StringIO()\n" +
                "sys.stderr.flush()\n" +
                "import os\n" +
                "os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'\n";

            PythonEngine.RunSimpleString(codeToRedirectOutput);
            PythonEngine.BeginAllowThreads();
        }
    }
}
