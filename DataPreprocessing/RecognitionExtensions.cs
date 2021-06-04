using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using FaceRecognitionDotNet;
using OpenCvSharp;

namespace DataPreprocessing
{
    public static class RecognitionExtensions
    {
        public static (Rect rect, IDictionary<FacePart, IEnumerable<FacePoint>> landmarks)? BiggestWithConsideredFaceParts(this Rect[] rects,
            List<IDictionary<FacePart, IEnumerable<FacePoint>>> landmarks, FacePart[] consideredFaceParts)
        {
            if (landmarks.Count == 0)
            {
                return null;
            }

            var selectedRects = new List<(Rect rect, IDictionary<FacePart, IEnumerable<FacePoint>> landmarks)>();

            //for each landmark that contains considered face parts
            foreach (var landmark in landmarks.Where(l => l.Keys.Intersect(consideredFaceParts).All(consideredFaceParts.Contains)))
            {
                foreach (var rect in rects)
                {
                    var contains = true;
                    foreach (var (k, v) in landmark.Where((kv) => consideredFaceParts.Contains(kv.Key)))
                    {
                        if (!v.All(p => rect.Contains(p.Point.X, p.Point.Y)))
                        {
                            contains = false;
                            break;
                        }
                    }

                    if (contains && !selectedRects.Select(v => v.rect).Contains(rect))
                    {
                        selectedRects.Add((rect, landmark));
                    }
                }
            }

            selectedRects.Sort((v1, v2) => v1.rect.Width * v1.rect.Height <= v2.rect.Width * v2.rect.Height ? -1 : 1);

            if (selectedRects.Count == 0)
            {
                return null;
            }

            return (selectedRects[^1]);
        }
    }
}
