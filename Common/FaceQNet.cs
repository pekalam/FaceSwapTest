using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Keras.Models;
using Numpy;
using OpenCvSharp;
using Python.Runtime;


namespace Common
{
    public class FaceQNet : IDisposable
    {
        private readonly BaseModel _model;
        private readonly Mat _photoRes;

        public FaceQNet(string modelPath)
        {
            _model = BaseModel.LoadModel(modelPath);
            _photoRes = Mat.Zeros(MatType.CV_8U, 224, 224, 3).ToMat();
        }

        public float GetScore(Mat photo)
        {
            if (photo.Cols != 224 || photo.Rows != 224)
            {
                Cv2.Resize(photo, _photoRes, new Size(224, 224));
                photo = _photoRes;
            }

            float[,,,] photoArr = new float[1, 224, 224, 3];

            for (int i = 0; i < photo.Cols; i++)
            {
                for (int j = 0; j < photo.Rows; j++)
                {
                    var v = photo.Get<Vec3b>(j, i);
                    photoArr[0, j, i, 0] = v[0];
                    photoArr[0, j, i, 1] = v[1];
                    photoArr[0, j, i, 2] = v[2];
                }
            }

            var npLock = PythonEngine.AcquireLock();

            NDarray numpyPhoto = np.array(photoArr);
            var result = _model.Predict(numpyPhoto);
            var score = result[0].asscalar<float>();
            result.Dispose();
            numpyPhoto.Dispose();

            PythonEngine.ReleaseLock(npLock);
            return score;
        }

        public void Dispose()
        {
            _photoRes.Dispose();
            _model.Dispose();
        }
    }
}
