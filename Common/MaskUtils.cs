using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using FaceRecognitionDotNet;
using OpenCvSharp;
using Point = OpenCvSharp.Point;

namespace Common
{
    public static class MaskUtils
    {
        public static Mat? GetMasked(Mat img, IDictionary<FacePart, IEnumerable<FacePoint>> landmark)
        {
            var lChin = landmark[FacePart.Chin]
                .First(v => v.Index == 0).Point;
            var c = new List<Point>();

            c.Add(new Point(lChin.X, lChin.Y));

            foreach (var el in landmark[FacePart.LeftEyebrow].Where(v => v.Index < 20))
            {
                c.Add(new Point(el.Point.X, el.Point.Y));
            }

            foreach (var el in landmark[FacePart.RightEyebrow].Where(v => v.Index >= 24))
            {
                c.Add(new Point(el.Point.X, el.Point.Y));
            }

            if (landmark[FacePart.Chin].Max(v => v.Point.Y) < 96 - 6)
            {
                return null;
            }

            foreach (var el in landmark[FacePart.Chin].Reverse())
            {
                var cpt = new Point(el.Point.X, el.Point.Y);
                c.Add(cpt);
            }

            var newImg = img.Clone();
            newImg.SetTo(Scalar.Black);
            Point[][] contours = new Point[][]
            {
                c.ToArray()
            };
            Cv2.DrawContours(newImg, contours, 0, Scalar.White, -1);
            img.CopyTo(newImg, newImg);


            return newImg;
        }
    }
}
