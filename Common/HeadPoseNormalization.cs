using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using FaceRecognitionDotNet;
using OpenCvSharp;
using Point = FaceRecognitionDotNet.Point;

namespace Common
{
    public class FaceRecognitionModelSettings
    {
        public string Location { get; set; } = ".";
    }

    public static class SharedFaceRecognitionModel
    {
        private static object _lck = new object();
        private static Task<FaceRecognition> _loadTask;


        public static void Init(FaceRecognitionModelSettings settings)
        {
            _loadTask = Task.Factory.StartNew<FaceRecognition>(() => FaceRecognition.Create(settings.Location),
                TaskCreationOptions.LongRunning);
        }

        public static List<FaceEncoding> FaceEncodingsSync(Image image,
            IEnumerable<Location> knownFaceLocation = null,
            int numJitters = 1,
            PredictorModel model = PredictorModel.Small)
        {
            lock (_lck)
            {
                return _loadTask.Result.FaceEncodings(image, knownFaceLocation, numJitters, model).ToList();
            }
        }

        public static Task<FaceRecognition> LoadTask => _loadTask;
        public static FaceRecognition Model => _loadTask.Result;
    }

    public class HeadPoseNormalization
    {
        private static readonly float[,] MINMAX_TEMPLATE = new[,]
        {
            {0.0f, 0.17856914f},
            {0.00412831f, 0.31259227f},
            {0.0196793f, 0.44770938f},
            {0.04809872f, 0.5800727f},
            {0.10028344f, 0.70349526f},
            {0.17999782f, 0.81208664f},
            {0.27627307f, 0.90467805f},
            {0.38463727f, 0.98006284f},
            {0.5073561f, 1.0f},
            {0.63014114f, 0.9761118f},
            {0.7386777f, 0.89921385f},
            {0.8354747f, 0.80513287f},
            {0.91434467f, 0.6945623f},
            {0.9643504f, 0.56826204f},
            {0.9887058f, 0.432444f},
            {0.9993123f, 0.29529294f},
            {1.0f, 0.15909716f},
            {0.09485531f, 0.07603313f},
            {0.15534875f, 0.02492465f},
            {0.2377474f, 0.01139098f},
            {0.32313403f, 0.02415778f},
            {0.4036699f, 0.05780071f},
            {0.56864655f, 0.0521157f},
            {0.65128165f, 0.01543965f},
            {0.7379608f, 0.0f},
            {0.82290924f, 0.01191543f},
            {0.88739765f, 0.06025707f},
            {0.48893312f, 0.15513189f},
            {0.48991537f, 0.24343018f},
            {0.49092147f, 0.33176517f},
            {0.49209353f, 0.422107f},
            {0.397399f, 0.48004663f},
            {0.4442625f, 0.49906778f},
            {0.4949509f, 0.5144414f},
            {0.54558265f, 0.49682876f},
            {0.59175086f, 0.47722608f},
            {0.194157f, 0.16926692f},
            {0.24600308f, 0.13693026f},
            {0.31000495f, 0.13735634f},
            {0.36378494f, 0.17794687f},
            {0.3063696f, 0.19082251f},
            {0.24390514f, 0.19138186f},
            {0.6189632f, 0.17277813f},
            {0.67249435f, 0.12988105f},
            {0.7362857f, 0.1279085f},
            {0.7888591f, 0.15817115f},
            {0.74115133f, 0.18155812f},
            {0.6791372f, 0.18370388f},
            {0.30711025f, 0.6418497f},
            {0.3759703f, 0.6109595f},
            {0.44670257f, 0.5970508f},
            {0.49721557f, 0.60872644f},
            {0.5500201f, 0.5954327f},
            {0.6233016f, 0.6070911f},
            {0.69541407f, 0.6341429f},
            {0.628068f, 0.70906836f},
            {0.5573954f, 0.7434471f},
            {0.50020397f, 0.7505844f},
            {0.44528747f, 0.74580276f},
            {0.37508208f, 0.7145425f},
            {0.3372878f, 0.64616466f},
            {0.44701463f, 0.64064664f},
            {0.49795204f, 0.6449633f},
            {0.5513943f, 0.6385937f},
            {0.6650228f, 0.63955915f},
            {0.5530556f, 0.67647934f},
            {0.4986481f, 0.68417645f},
            {0.44657204f, 0.6786047f}
        };

        private Image ToImage(Mat photo)
        {
            var bytes = new byte[photo.Rows * photo.Cols * photo.ElemSize()];
            Marshal.Copy(photo.Data, bytes, 0, bytes.Length);

            var img = FaceRecognition.LoadImage(bytes, photo.Rows, photo.Cols, photo.ElemSize(), Mode.Rgb);
            return img;
        }

        private List<IDictionary<FacePart, IEnumerable<FacePoint>>> GetLandmarks(Mat src)
        {
            var landmarks = SharedFaceRecognitionModel.Model.FaceLandmark(ToImage(src)).ToList();
            return landmarks;
        }

        public (Mat, Point2f[] p1, Point2f[] p2) NormalizePosition(Mat src, Rect faceRect,
            List<IDictionary<FacePart, IEnumerable<FacePoint>>> landmarks = null)
        {
            if (landmarks == null)
            {
                landmarks = GetLandmarks(src);
            }

            var points = new List<Point>();
            foreach (var landmark in landmarks.First())
            {
                foreach (var p in landmark.Value.ToArray())
                {
                    points.Add(p.Point);
                }
            }

            var innerEyesAndBottomLip = new int[3];

            var bml = landmarks.First()[FacePart.BottomLip].First(v =>
                v.Point.X == landmarks.First()[FacePart.BottomLip].ElementAt(0).Point.X);
            innerEyesAndBottomLip[2] = Enumerable.Select(points, (p, i) => new {p, i})
                .Where((v, i) => v.p.X == bml.Point.X && v.p.Y == bml.Point.Y)
                .Select(v => v.i).First();


            var el = landmarks.First()[FacePart.LeftEye]
                .First(v => v.Point.X == landmarks.First()[FacePart.LeftEye].Min(p => p.Point.X));
            innerEyesAndBottomLip[0] = Enumerable.Select(points, (p, i) => new {p, i})
                .Where((v, i) => v.p.X == el.Point.X && v.p.Y == el.Point.Y)
                .Select(v => v.i).First();

            var er = landmarks.First()[FacePart.RightEye]
                .First(v => v.Point.X == landmarks.First()[FacePart.RightEye].Max(p => p.Point.X));
            innerEyesAndBottomLip[1] = Enumerable.Select(points, (p, i) => new {p, i})
                .Where((v, i) => v.p.X == er.Point.X && v.p.Y == er.Point.Y)
                .Select(v => v.i).First();


#if DEV_MODE
            foreach (var i in innerEyesAndBottomLip)
            {
                Cv2.Ellipse(src,
                    new RotatedRect(new Point2f(points[i].X, points[i].Y), new Size2f(5, 5), 0),
                    Scalar.Red);
            }
#endif


            var p1 = points.Where((p, i) => innerEyesAndBottomLip.Contains(i));
            var p2 = new[]
            {
                new Point2f(MINMAX_TEMPLATE[innerEyesAndBottomLip[0], 0],
                    MINMAX_TEMPLATE[innerEyesAndBottomLip[0], 1]),
                new Point2f(MINMAX_TEMPLATE[innerEyesAndBottomLip[1], 0],
                    MINMAX_TEMPLATE[innerEyesAndBottomLip[1], 1]),
                new Point2f(MINMAX_TEMPLATE[innerEyesAndBottomLip[2], 0],
                    MINMAX_TEMPLATE[innerEyesAndBottomLip[2], 1]),
            };
            var p1Fc = p1.Select(p =>
                new Point2f((p.X),
                    (p.Y))).ToArray();
            var p2Fc = p2.Select(p => new Point2f(p.X * faceRect.Width, p.Y * faceRect.Height)).ToArray();

            var h = Cv2.GetAffineTransform(p1Fc,p2Fc);

            var ret = new Mat();
            Cv2.WarpAffine(src, ret, h, new Size(faceRect.Width, faceRect.Height), InterpolationFlags.Linear,
                BorderTypes.Constant);
            h.Dispose();

            return (ret, p1Fc, p2Fc);
        }
    }
}