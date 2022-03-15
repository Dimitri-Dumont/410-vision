package edu.pdx.cs410j.objectdect;
import androidx.appcompat.app.AppCompatActivity;
import android.content.Context;
import android.content.res.AssetManager;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.widget.Toast;
import org.opencv.android.*;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.dnn.Net;
import org.opencv.dnn.Dnn;
import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;


public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {
    private static String TAG = "MainActivity";

    static {
        if (!OpenCVLoader.initDebug())
            Log.e("OpenCV", "Unable to load OpenCV!");
        else
            Log.d("OpenCV", "OpenCV loaded Successfully!");
    }

    JavaCameraView jc;
    Mat mat1;
    Mat mat2;
    private Net net;
    Names names = new Names();
    String[] classNames = names.classNames;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_main);

        jc = (JavaCameraView) findViewById(R.id.Cam);
        jc.setVisibility(SurfaceView.VISIBLE);
        jc.setCvCameraViewListener(this);
        jc.enableView();
        jc.setCameraPermissionGranted();

        try {

            String proto = getPath("MobileNetSSD_deploy.prototxt", this);
            String weights = getPath("MobileNetSSD_deploy.caffemodel", this);

            net = Dnn.readNetFromCaffe(proto, weights);
        } catch (Exception e) {
            Log.e(e.getMessage(), "poop");
        }


        BaseLoaderCallback bs = new BaseLoaderCallback(this) {
            @Override
            public void onManagerConnected(int status) {
                switch (status) {
                    case BaseLoaderCallback.SUCCESS:
                        jc.enableView();
                    default:
                        super.onManagerConnected(status);
                        Log.e("NOOOOOOOOOO", "God why");
                        break;
                }
            }
        };
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mat1 = new Mat(width, height, CvType.CV_8UC4);
        mat2 = new Mat(width, height, CvType.CV_8UC4);
    }

    @Override
    public void onCameraViewStopped() {
        mat1.release();
        mat2.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        final int w = 300;
        final int h = 300;
        final double scale = 0.007843;
        final double mean = 127.5;
        final double thres = 0.2;

        Mat frame = inputFrame.rgba();
        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB);

        Mat ip = Dnn.blobFromImage(frame, scale,
                new Size(w, h),
                new Scalar(mean, mean, mean), /*swapRB*/false, /*crop*/false);
        net.setInput(ip);

        Mat objects = net.forward();
        int cols = frame.cols();
        int rows = frame.rows();
        objects = objects.reshape(1, (int) objects.total() / 7);
        for (int i = 0; i < objects.rows(); ++i) {
            double confidence = objects.get(i, 2)[0];
            if (confidence > thres) {
                int classId = (int) objects.get(i, 1)[0];
                int left = (int) (objects.get(i, 3)[0] * cols);
                int top = (int) (objects.get(i, 4)[0] * rows);
                int right = (int) (objects.get(i, 5)[0] * cols);
                int bottom = (int) (objects.get(i, 6)[0] * rows);
                //Draw bounding box
                Imgproc.rectangle(frame, new Point(left, top), new Point(right, bottom),
                        new Scalar(0, 255, 0));
                String label = classNames[classId] + ": " + confidence;
                int[] baseLine = new int[1];
                Size labelSize = Imgproc.getTextSize(label, 0, 0.5, 1, baseLine);
                Imgproc.rectangle(frame, new Point(left, top - labelSize.height),
                        new Point(left + labelSize.width, top + baseLine[0]),
                        new Scalar(255, 255, 255), Imgproc.FILLED);
                Imgproc.putText(frame, label, new Point(left, top),
                        0, 0.5, new Scalar(0, 0, 0));
            }
        }
        return frame;
    }

    @Override
    public void onPause() {
        super.onPause();
        if (jc != null)
            jc.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug())
            Toast.makeText(getApplicationContext(), "OpenCV issue", Toast.LENGTH_SHORT).show();
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (jc != null)
            jc.disableView();
    }

    private static String getPath(String file, Context context) {
        AssetManager assetManager = context.getAssets();
        BufferedInputStream inputStream = null;
        try {
            inputStream = new BufferedInputStream(assetManager.open(file));
            byte[] data = new byte[inputStream.available()];
            inputStream.read(data);
            inputStream.close();
            File outFile = new File(context.getFilesDir(), file);
            FileOutputStream os = new FileOutputStream(outFile);
            os.write(data);
            os.close();
            return outFile.getAbsolutePath();
        } catch (IOException ex) {
            return null;
        }
    }
}